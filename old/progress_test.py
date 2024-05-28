import time

def dummy_func(progress):
    task1 = progress.add_task("[red]Downloading...", total=10)
    for i in range(10):
        time.sleep(0.5)
        progress.update(task1,advance=1,refresh=True)
        
def dummy_func2(progress):
    print('hi')