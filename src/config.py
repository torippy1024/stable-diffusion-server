from pydantic import BaseSettings


class Settings(BaseSettings):
    diffusion_mode: str = "wd"


settings = Settings()
