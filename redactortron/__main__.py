"""Allow ``python -m redactortron …`` when console scripts are not on PATH."""

from redactortron.cli import main

if __name__ == "__main__":
    main()
