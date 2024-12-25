import importlib
import inspect
import logging
import pkgutil
from typing import List, Type, Callable, Any

import json_repair as jr
import json5

logger = logging.getLogger(__name__)


class JsonHelper:

    @staticmethod
    def fix_malformed_json(json_str: str) -> str:
        """
        Try and fix the syntax error(s) in a JSON string.

        :param json_str: The input JSON string.
        :return: The fixed JSOn string.
        """

        return jr.repair_json(json_str, skip_json_loads=True)

    @staticmethod
    def load_and_fix_json(json_str: Any) -> dict:
        parsed_data = {}
        if not isinstance(json_str, str):
            return json_str

        try:
            parsed_data = json5.loads(json_str)
        except ValueError:
            logger.warning("Caught ValueError: trying again after repairing JSON...")
            try:
                parsed_data = json5.loads(JsonHelper.fix_malformed_json(json_str))
            except ValueError:
                logger.error("Caught ValueError: failed to parse JSON!")

        except RecursionError:
            logger.error("Caught RecursionError while parsing JSON. Cannot generate the slide deck!")

        except Exception:
            logger.error("Caught ValueError: failed to parse JSON!")

        return parsed_data


class AutoDiscover:

    @classmethod
    def discover_subclasses(
        cls,
        base_class: Type,
        register: Callable,
        base_packages: List[str],
        max_depth: int = 3,
    ):
        """
        Recursively scan packages and subpackages for subclasses of a specified base class

        Args:
            base_class (Type, optional): Base class to discover subclasses of.
            register (Callable[[Type], None], optional): Custom registration function.
            base_packages (List[str]): List of base package names to start discovery
            max_depth (int): Maximum recursion depth to prevent infinite loops, default 3
        """
        discovered_subclasses = set()

        def discover_in_package(package_name: str, current_depth: int = 0):
            if current_depth > max_depth:
                return

            try:
                package = importlib.import_module(package_name)

                # Scan for classes in current module
                for _, obj in inspect.getmembers(package):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, base_class)
                        and obj is not base_class
                        and obj not in discovered_subclasses
                    ):
                        register(obj)
                        discovered_subclasses.add(obj)

                #  Recursively discover in submodules
                for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
                    full_module_name = f"{package_name}.{name}"

                    # Discover classes in the current module
                    try:
                        module = importlib.import_module(full_module_name)
                        for _, obj in inspect.getmembers(module):
                            if (
                                inspect.isclass(obj)
                                and issubclass(obj, base_class)
                                and obj is not base_class
                                and obj not in discovered_subclasses
                            ):
                                register(obj)
                                logger.info(
                                    "Discovered and registered subclasses: %s of base class %s",
                                    obj.__name__,
                                    base_class.__name__,
                                )
                                discovered_subclasses.add(obj)
                    except ImportError as e:
                        logger.warn(f"Could not import module {full_module_name}: {e}")

                    # If it's a package, recursively discover
                    if is_pkg:
                        discover_in_package(full_module_name, current_depth + 1)

            except ImportError as e:
                logger.warn(f"Could not discover agents in {package_name}: {e}")

        # Start discovery for each base package
        for base_package in base_packages:
            discover_in_package(base_package)

        return discovered_subclasses
