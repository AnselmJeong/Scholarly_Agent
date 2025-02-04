from tomli import load

with open("config.toml", "rb") as f:
    CONFIG = load(f)
