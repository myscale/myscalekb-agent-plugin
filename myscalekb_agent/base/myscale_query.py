import logging
import os
import re
import time
from typing import List

import nanoid
from clickhouse_connect.driver import Client
from clickhouse_connect.driver.exceptions import OperationalError
from pydantic import BaseModel, Field

from myscalekb_agent.base.schemas.knowledge_base import KnowledgeScope

logger = logging.getLogger(__name__)


class QueryParams(BaseModel):
    alpha: float | None = Field(default=1.0)
    limit: int | None = Field(default=10)
    offset: int | None = Field(default=0)


class MyScaleQuery:

    database: str = os.getenv("MYSCALE_DATABASE", "kb")

    @staticmethod
    def query(client: Client, q_str: str, timeout: int = 30, run_command: bool = False):
        query_id = nanoid.generate(size=6)

        # pylint: disable=unused-variable
        def print_summary_sql(q_str, query_id):
            from_index = q_str.lower().find("from")

            if from_index != -1:
                start = max(from_index - 50, 0)
                end = from_index + len("from") + 50

                from_context = q_str[start:end]
                if not from_context.strip().lower().startswith("select"):
                    from_context = "..." + from_context
            else:
                from_context = "(FROM keyword not found)"
            logger.info("Querying MyScale %s, debug sql: %s...", query_id, from_context)

        debug_sql = bool(os.getenv("MYSCALE_DEBUG_SQL", "False"))
        if debug_sql:
            logger.info("MyScale Query - %s start, debug sql: %s", query_id, q_str)
        else:
            print_summary_sql(q_str, query_id)

        retries = 3
        attempt = 0
        settings = {"max_execution_time": timeout}
        while attempt < retries:
            try:
                start_time = time.time()
                if run_command:
                    res = client.command(q_str, settings=settings)
                else:
                    res = client.query(q_str, settings=settings)
                duration = time.time() - start_time
                logger.info("MyScale Query - %s end, duration: %.3f seconds", query_id, duration)
                return res
            except OperationalError as e:
                logger.warning(f"Attempt {attempt + 1} failed with a timeout: {e}. Retrying...")
                attempt += 1
                time.sleep(0.1)

        raise Exception(f"Failed to query myscale after {retries} attempts.")

    @staticmethod
    def gen_where_str(knowledge_scopes: List[KnowledgeScope], skip_only_kb: bool = False) -> str:
        where_str = ""
        where_keyword = "WHERE"

        # group kb_id first
        only_kb = [scope.kb_id for scope in knowledge_scopes if not scope.doc_ids]
        if only_kb and not skip_only_kb:
            where_str += f"{where_keyword} (kb_id IN {only_kb}) "
            where_keyword = "OR"

        for scope in knowledge_scopes:
            if scope.doc_ids:
                where_str += f"{where_keyword} (kb_id = '{scope.kb_id}' AND doc_id IN {scope.doc_ids})"
                where_keyword = "OR"

        # if where is empty, generate false condition
        if not where_str:
            where_str = f"WHERE 1=2"

        return where_str

    @staticmethod
    def text_escape(text: str) -> str:
        escaping = [
            "-",
            "+",
            "^",
            "`",
            ":",
            "{",
            "}",
            '"',
            "[",
            "]",
            "(",
            ")",
            "~",
            "!",
            "\\",
            "*",
        ]
        pattern = "[" + re.escape("".join(escaping)) + "]"
        escaping_text = re.sub(pattern, " ", text)
        return escaping_text.replace("'", "\\'")
