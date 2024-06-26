from rich.console import Console
from rich.progress import Progress
import sys
import time


with open('logfile_test.log', 'w') as log_file:
    console = Console(file=log_file, force_terminal=True)  


    with Progress(console=console) as progress:

        task1 = progress.add_task("[red]Downloading...", total=1000)
        task2 = progress.add_task("[green]Processing...", total=1000)
        task3 = progress.add_task("[cyan]Cooking...", total=1000)

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.3)
            progress.update(task3, advance=0.9)
            time.sleep(0.02)