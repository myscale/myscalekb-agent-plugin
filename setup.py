from setuptools import setup, find_namespace_packages

setup(
    name="myscalekb-agent-base",
    version="0.0.1",
    author="jiachengs",
    author_email="jiachengs@myscale.com",
    description="Base package for myscalekb agent",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/myscale/myscalekb-agent-plugin",

    packages=find_namespace_packages(include=['myscalekb_agent.*']),

    package_data={
        "myscalekb_agent.base": ["*.txt", "*.json"],
    },

    namespace_packages=['myscalekb_agent'],

    python_requires=">=3.12",

    install_requires=[
        # app required
        # langchain family
        'langchain==0.3.2',
        'langchain-openai==0.2.2',
        'langchain-community==0.3.1',
        'langgraph==0.2.35',

        # others
        'nanoid~=2.0.0',
        'clickhouse-connect~=0.7.18',
        'json5~=0.9.25',
        'json_repair~=0.30.0',
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
)
