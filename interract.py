import itertools
import threading
import time
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from functools import wraps


class ParserOptions:
    def __init__(self, 
                 long:str, 
                 short:str, 
                 convert:type|function=None,
                 default:str = None, 
                 choices:list[str]=None, 
                 help:str=None, 
                 required=False
                 ) -> None:
        """A custom class to help manage the parsing of input for python script and use them as arguments.
        
        Args:
            long (str): A letter or two for arg.
            short (str,optional): The actual name of the arg. Defaults to None.
            convert (type|function, optional): Automatically convert an argument to the given type.
            default (str, optional): A default arg to give. Defaults to None.
            choices (list[any], optional): A list of choices the args that can be used. Default to None.
            help (str, optional): Help to better understand the use of the arg. Defaults to ''.
            required (bool, optional): Indicate whether an argument is required or optional
        """
        self.short = short
        self.long = long
        self.type = convert
        self.default = default
        self.choices = choices
        self.help = help
        self.required = required

def parse_input(
        parser_config:ParserOptions|list[ParserOptions]=None, 
        prog_name = "",
        descr = "",
        epilog = "",
        mode='default'
        )-> dict:
    """An easier way to parse the inputs of python script

    Args:
        parser_config (ParserOptions|list[ParserOptions]): _description_. Defaults to None.
        mode (str, optional): Help formatter mode. Defaults to 'default'.

    Returns:
        _type_: _description_
    """
    # TODO finish to deal with the parser
    if mode == 'default':
        parser = ArgumentParser(
            prog=prog_name,
            description=descr,
            epilog=epilog,
            formatter_class=ArgumentDefaultsHelpFormatter)
    else :
        raise NotImplementedError("The non default case has not been implemented yet")
    if parser_config:
        if type(parser_config) == ParserOptions:
            parser.add_argument('-'+parser_config.short, 
                    '--'+parser_config.long, 
                    default=parser_config.default,
                    help=parser_config.help)
        elif type(parser_config) == list:
            for parse_arg in parser_config:
                parser.add_argument('-'+parse_arg.short, 
                                    '--'+parse_arg.long, 
                                    default=parse_arg.default,
                                    help=parse_arg.help)
        return vars(parser.parse_args())
    else:
        raise ValueError("Need to give ParserOptions or a list of ParserOptions")

def animate(stop_event, messages):
    """Animate a spinner in the console with dynamic loading text.

    Args:
    - stop_event (threading.Event): Event to signal when to stop the spinner.
    - messages (list): List of messages to display in sequence along with the spinner.
    """
    for message, c in zip(itertools.cycle(messages), itertools.cycle(['|', '/', '-', '\\'])):
        if stop_event.is_set():
            break
        sys.stdout.write(f'\r{message} {c}')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write(f'\rDone! \n')

def spinner_decorator(messages):
    """Decorator to show a spinner with dynamic loading text while a function is running.

    Args:
    - messages (list): List of messages to display in sequence along with the spinner.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an event to control the spinner
            stop_event = threading.Event()
            # Start spinner thread
            t = threading.Thread(target=animate, args=(stop_event, messages))
            t.start()
            try:
                # Run the actual function
                result = func(*args, **kwargs)
            finally:
                # Stop the spinner
                stop_event.set()
                # Ensure the spinner thread finishes
                t.join()
            return result
        return wrapper
    return decorator

def dynamic_loading_bar(message="loading", total = 100):
    """Decorator to show a dynamic loading bar while a function is running.

    Args:
    - message (str): Message to display along with the loading bar.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create an event to control the loading bar
            stop_event = threading.Event()
            progress_bar = tqdm(total=total, desc=message, ncols=100, leave=False, bar_format='{desc}: {percentage:3.0f}%|{bar}| {remaining_s}s')

            def update_progress():
                """Update the progress bar while the function runs."""
                for _ in itertools.cycle([0]):
                    if stop_event.is_set():
                        break
                    progress_bar.update(1)
                    time.sleep(0.1)
                    progress_bar.n = (progress_bar.n + 1) % 100
                    progress_bar.last_print_t = time.time()
                    progress_bar.refresh()

            # Start the progress bar thread
            t = threading.Thread(target=update_progress)
            t.start()
            try:
                # Run the actual function
                result = func(*args, **kwargs)
            finally:
                # Stop the loading bar
                stop_event.set()
                # Ensure the loading bar thread finishes
                t.join()
                progress_bar.close()
            return result
        return wrapper
    return decorator

