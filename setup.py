from setuptools import setup, find_namespace_packages

setup(
    name="myscalekb-agent-base",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
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
    
    install_requires=[],
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.12",
    ],
)