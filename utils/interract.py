import itertools
import threading
import time
import sys, warnings, logging
import argparse
from tqdm import tqdm
from functools import wraps
from typing import Union, List, Any
from copy import deepcopy
import numpy as np
import algorithms as alg

# ONLY PLACE WHERE TO ADD NEW ALGO NAMES
def build_nuc_algos():
    """
    Build a dictionary mapping SBNUC algorithm names to their corresponding functions.

    This function creates and returns a dictionary where the keys are the names of different
    Scene-Based Non-Uniformity Correction (SBNUC) algorithms, and the values are the functions
    that implement these algorithms.

    Returns:
        dict: A dictionary mapping SBNUC algorithm names (str) to their corresponding functions.
              The available algorithms are:
              -> 'SBNUCIRFPA': Function for SBNUCIRFPA algorithm
              -> 'AdaSBNUCIRFPA': Function for adaptive SBNUCIRFPA algorithm
              -> 'AdaSBNUCIRFPA_reg': Function for adaptive SBNUCIRFPA with registration
              -> 'CstStatSBNUC': Function for constant statistics SBNUC
              -> 'SBNUCLMS': Function for SBNUCLMS algorithm
              -> 'SBNUCif_reg': Function for SBNUC with interframe registration
              -> 'AdaSBNUCif_reg': Function for adaptive SBNUC with interframe registration
              -> 'CompTempNUC' : Function to compensate Temperature variations through NUC
              -> 'NUCnlFilter' : Function to apply a non linear filter to the NUC
              -> 'RobustNUCIRFPA' : Function to apply a robust NUC on IRFPA
              -> 'AdaRobustNUCIRFPA' : Function to apply a robust NUC on IRFPA with adaptation
              -> 'SBNUC_smartCam_pipeA' : Function to apply a NUC smart camera algorithm using pipeline A
              -> 'SBNUC_smartCam_pipeB' : Function to apply a NUC smart camera algorithm using pipeline B
              -> 'SBNUC_smartCam_pipeC' : Function to apply a NUC smart camera algorithm using pipeline C
              -> 'SBNUCcomplement' : Function to apply a complement to the first filter
    """
    # TODO add new algos when implementation, only place with __init__.py
    return {
        'SBNUCIRFPA': alg.AdaSBNUCIRFPA.SBNUCIRFPA,                # Function for SBNUCIRFPA algorithm
        'AdaSBNUCIRFPA': alg.AdaSBNUCIRFPA.AdaSBNUCIRFPA,          # Function for adaptive SBNUCIRFPA algorithm
        'AdaSBNUCIRFPA_reg': alg.AdaSBNUCIRFPA.AdaSBNUCIRFPA_reg,  # Function for adaptive SBNUCIRFPA with registration
        'CstStatSBNUC': alg.SBNUCrgGLMS.CstStatSBNUC,            # Function for constant statistics SBNUC
        'SBNUCLMS': alg.SBNUCrgGLMS.SBNUCLMS,                    # Function for SBNUCLMS algorithm
        'SBNUCif_reg': alg.SBNUCif_reg.SBNUCif_reg,              # Function for SBNUC with interframe registration
        'AdaSBNUCif_reg': alg.SBNUCif_reg.AdaSBNUCif_reg,        # Function for adaptive SBNUC with interframe registration
        # 'CompTempNUC': alg.ComptTempNUC.CompTempNUC,              # Function to compensate Temperature variations through NUC
        'NUCnlFilter': alg.NUCnlFilter.NUCnlFilter,            # Function to apply a non linear filter to the NUC
        'RobustNUCIRFPA': alg.RobustNUCIRFPA.RobustNUCIRFPA,        # Function to apply a robust NUC on IRFPA
        'AdaRobustNUCIRFPA': alg.RobustNUCIRFPA.AdaRobustNUCIRFPA,  # Function to apply a robust NUC on IRFPA with adaptation
        'SBNUC_smartCam_pipeA': alg.SBNUC_smartCam.SBNUC_smartCam_pipeA,  # Function to apply a NUC smart camera algorithm using pipeline A
        'SBNUC_smartCam_pipeB': alg.SBNUC_smartCam.SBNUC_smartCam_pipeB,  # Function to apply a NUC smart camera algorithm using pipeline B
        'SBNUC_smartCam_pipeC': alg.SBNUC_smartCam.SBNUC_smartCam_pipeC,  # Function to apply a NUC smart camera algorithm using pipeline C
        'SBNUC_smartCam_own_pipe': alg.SBNUC_smartCam.SBNUC_smartCam_own_pipe, 
        'SBNUCcomplement': alg.SBNUCcomplement.SBNUCcomplement,       # Function to apply a complement to the first filter
        'morgan': alg.morgan.morgan,
        'morgan_moving': alg.morgan.morgan_moving,
        'morgan_filt' : alg.morgan.morgan_filt,
        'morgan_filt_haut' : alg.morgan.morgan_filt_haut,
        'zac_NUCnlFilter' : alg.zac.zac_NUCnlFilter,
        'zac_smartCam' : alg.zac.zac_smartCam,
        'zac_AdaSBNUCIRFPA_window' : alg.zac.zac_AdaSBNUCIRFPA_window,
        # 'zac_AdaSBNUCIRFPA_mean' : alg.zac.zac_AdaSBNUCIRFPA_mean,
        'zac_RobustNUCIRFPA' : alg.zac.zac_RobustNUCIRFPA,
        'zac_SBNUCcomplement' : alg.zac.zac_SBNUCcomplement,
        'zac_CstStatSBNUC' : alg.zac.zac_CstStatSBNUC,
        'zac_SBNUCLMS' : alg.zac.zac_SBNUCLMS,
        'zac_AdaSBNUCif_reg' : alg.zac.zac_AdaSBNUCif_reg,
    }


def set_logging_info(mode='default') -> None:
    """
    Set up logging configuration for the application.

    Args:
        mode (str, optional): The mode of logging. Defaults to 'default'.
                              Currently, only 'default' mode is supported.

    Returns:
        None
    """
    if mode == 'default':
        # Configure logging with INFO level and a simple format
        logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
        
        # Ignore warnings to avoid cluttering the output
        warnings.simplefilter("ignore")


class ParserOptions:
    def __init__(self, 
                 long: str, 
                 short: str = None, 
                 action: str = None,
                 choices: List[Any] = None,
                 const: Any = None,
                 default: Any = None, 
                 dest: str = None,
                 help: str = '', 
                 metavar: str = '',
                 nargs: Union[int, str] = None,
                 required: bool = False,
                 type: Union[type, Any] = None
                 ) -> None:
        """A custom class to help manage the parsing of input for python script and use them as arguments.

        Args:
            long (str): The actual name of the arg.
            short (str, optional): A letter or two for arg. Defaults to None.
            action (str, optional): The basic type of action to be taken when this argument is encountered at the command line.
            choices (list[any], optional): A list of choices the args that can be used. Default to None.
            const (any, optional): A constant value required by some action and nargs selections.
            default (Any, optional): A default arg to give. Defaults to None.
            dest (str, optional): The name of the attribute to be added to the object returned by parse_args(). Defaults to None.
            help (str, optional): Help to better understand the use of the arg. Defaults to ''.
            metavar (str, optional): A name for the argument in usage messages. Defaults and prefered to ''.
            nargs (int or str, optional): The number of command-line arguments that should be consumed. Defaults to None.
            required (bool, optional): Indicate whether an argument is required or optional. Defaults to False.
            type (type or function, optional): Automatically convert an argument to the given type. Defaults to None.
        """
        self.long = long
        self.short = short
        self.action = action
        self.choices = choices
        self.const = const
        self.default = default
        self.dest = dest
        self.help = help
        # means the input doesn't need an argument => no metavar
        if action == "store_true":
            self.metavar = None
        else :
            self.metavar = metavar
        self.nargs = nargs
        self.required = required
        self.type = type
    
    def __str__(self) -> str:
        return f"ParserOptions(long={self.long}, short={self.short}, action={self.action}, choices={self.choices}, const={self.const}, default={self.default}, dest={self.dest}, help={self.help}, metavar={self.metavar}, nargs={self.nargs}, required={self.required}, type={self.type}"
    
    def __len__(self) -> int:
        return len(self.__dict__)
    
    def __eq__(self, o: object) -> bool:
        return self.len() == o.len()
    
    def __ne__(self, o: object) -> bool:
        return self.len() != o.len()
    
    def __lt__(self, o: object) -> bool:
        return self.len() < o.len()
    
    def __le__(self, o: object) -> bool:
        return self.len() <= o.len()
    
    def __gt__(self, o: object) -> bool:
        return self.len() > o.len()
    
    def __ge__(self, o: object) -> bool:
        return self.len() >= o.len()
    
    def __hash__(self) -> int:
        return hash(self.__dict__)
    
    def __getitem__(self, key: str) -> str:
        return self.__dict__[key]
    
    def __setitem__(self, key: str, value: str) -> None:
        self.__dict__[key] = value

    def __delitem__(self, key: str) -> None:
        del self.__dict__[key]

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__
    
    def __iter__(self):
        return iter(self.__dict__)
    
    def __copy__(self):
        return ParserOptions(
            self.long, 
            self.short, 
            self.action, 
            self.choices, 
            self.const, 
            self.default,
            self.dest, 
            self.help, 
            self.metavar, 
            self.nargs, 
            self.required, 
            self.type
        )
        
    def __deepcopy__(self, memo):
        """Create a deep copy of this ParserOptions instance."""
        return ParserOptions(
            long=self.long,
            short=self.short,
            action=self.action,
            choices=deepcopy(self.choices, memo),
            const=deepcopy(self.const, memo),
            default=deepcopy(self.default, memo),
            dest=self.dest,
            help=self.help,
            metavar=self.metavar,
            nargs=self.nargs,
            required=self.required,
            type=self.type
        )


def parse_input(parser_config: Union[ParserOptions, List[ParserOptions]] = None, 
                prog_name: str = "",
                descr: str = "",
                epilog: str = "",
                mode: str = 'default'
                ) -> dict[str, Any]:
    """An easier way to parse the inputs of python script.

    Args:
        parser_config (ParserOptions or list[ParserOptions]): Configuration for the parser. Defaults to None.
        prog_name (str, optional): Program name. Defaults to "".
        descr (str, optional): Program description. Defaults to "".
        epilog (str, optional): Epilog description. Defaults to "".
        mode (str, optional): Help formatter mode. Defaults to 'default'.

    Returns:
        dict: A dictionary with the input parsed.
    """
    if mode == 'default':
        parser = argparse.ArgumentParser(
            prog=prog_name,
            description=descr,
            epilog=epilog,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    else:
        raise NotImplementedError("The non-default case has not been implemented yet")
    
    if parser_config:
        if isinstance(parser_config, ParserOptions):
            add_argument_to_parser(parser, parser_config)
        elif isinstance(parser_config, list):
            for option in parser_config:
                add_argument_to_parser(parser, option)
        else:
            raise ValueError("parser_config must be ParserOptions or a list of ParserOptions")
    else:
        raise ValueError("Need to give ParserOptions or a list of ParserOptions")
    
    return vars(parser.parse_args())

def add_argument_to_parser(parser, option: ParserOptions):
    """Helper function to add an argument to the parser."""
    args = {
        'action': option.action,
        'choices': option.choices,
        'const': option.const,
        'default': option.default,
        'dest': option.dest,
        'help': option.help+'.',
        'metavar': option.metavar,
        'nargs': option.nargs,
        'required': option.required,
        'type': option.type
    }

    # add choices in the help text without too much verbose
    if args['choices']:
        args['help'] += f"\n(choices : {args['choices']})"
    
    # Remove None values from args dictionary
    args = {key: value for key, value in args.items() if value is not None}
    
    if option.short:
        parser.add_argument(f'-{option.short}', f'--{option.long}', **args)
    else:
        parser.add_argument(f'--{option.long}', **args)

def build_args_show_result():
    """
    Build and parse command-line arguments for the program.

    This function creates a dict of parser options for the command-line arguments
    required by the program. The arguments are parsed and returned.

    Arguments : folder_path, framerate

    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    parser_options = []  # Initialize an empty list to store parser options

    # Add parser option for the folder path containing binary files of the video
    parser_options.append(ParserOptions(
        long="folder_path",
        short="p",
        type=str,
        help="The path to the folder containing the video and the logs",
        required=True  # This argument is required
    ))

    # Add parser option for the framerate per second
    parser_options.append(ParserOptions(
        long="framerate",
        short="fps",
        type=float,
        default=30,  # Default framerate is set to 60 FPS
        help="The framerate per second for displaying the video"
    ))

    # Parse the input arguments using the defined parser options
    args = parse_input(
        parser_config=parser_options,
        prog_name="IR SBNUC prototypes"  # Program name for the parser
    )

    print(" --- Input parsed --- ")  # Indicate that input has been successfully parsed
    return args  # Return the parsed arguments

def build_args():
    """
    Build and parse command-line arguments for the program.

    This function creates a dict of parser options for the command-line arguments
    required by the program. The arguments are parsed and returned.

    Arguments : folder_path, width, height, depth, num_frames, framerate
    Flags : show_video

    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    parser_options = []  # Initialize an empty list to store parser options

    # Add parser option for the folder path containing binary files of the video
    parser_options.append(ParserOptions(
        long="folder_path",
        short="p",
        type=str,
        help="The path to the folder containing all the binary files of the video",
        required=True  # This argument is required
    ))

    # Add parser option for the folder path where to save results
    parser_options.append(ParserOptions(
        long="save_folder",
        short="save",
        type=str,
        help="The path to the folder containing all results from this algorithm",
    ))

    # Add parser option for the width of the video
    parser_options.append(ParserOptions(
        long="width",
        short="w",
        type=int,
        default=640,  # Default width is set to 640
        help="The width of the video",
    ))

    # Add parser option for the height of the video
    parser_options.append(ParserOptions(
        long="height",
        short="he",
        type=int,
        default=480,  # Default height is set to 480
        help="The height of the video",
    ))

    # Add parser option for the bit depth of the video
    parser_options.append(ParserOptions(
        long="depth",
        short="d",
        type=str,
        default="8b",  # Default bit depth is set to 8 bits
        help="The bits' depth of the video",
        choices=['8b', '14b', '16b', '32b', '64b', '128b', '256b'],  # Allowed choices for bit depth
    ))

    # TODO fix pb of helper otput when declaring two flags
    # Add parser option for showing the video or not
    parser_options.append(ParserOptions(
        long="show_video",
        short="s",
        action="store_true",  # This makes it a boolean flag
        help="Show the video if this flag is set",
    ))
    
    parser_options.append(ParserOptions(
        long="clean", 
        short="c",
        action="store_true", # This makes it a boolean flag
        help="Flag to say if frames are clean or not", 
    ))

    # Add parser option for the number of frames to compute
    parser_options.append(ParserOptions(
        long="num_frames",
        short="n",
        type=int,
        default=np.inf,  # Default is None, which means process all frames
        help="The number of frames to compute"
    ))

    # Add parser option for the framerate per second
    parser_options.append(ParserOptions(
        long="framerate",
        short="fps",
        type=float,
        default=30,  # Default framerate is set to 60 FPS
        help="The framerate per second for displaying the video"
    ))

    # Add parser option for the begining of a stable frame
    parser_options.append(ParserOptions(
        long="stable_frame",
        short="st",
        type=int,
        default=0,
        help="The begining of a stable scene"
    ))

    # Add parser option for the kernel size
    parser_options.append(ParserOptions(
        long="kernel_size",
        short="k",
        type=int,
        default=3,  # Default framerate is set to 3 FPS
        help="The size of the kernel to use for filtering. Must be an odd number"
    ))

    # Add parser option for the nuc adaptation algorithm
    nuc_algos = [alg for alg in build_nuc_algos().keys()]
    nuc_algos.append("all")
    parser_options.append(ParserOptions(
        long="nuc_algorithm", 
        short="nuc", 
        type=str, 
        default="morgan",
        nargs='+', 
        choices=nuc_algos, 
        help="Algorithms to use for nuc adaptation (can specify multiple)" 
    ))

    # Add parser option for the motion estimation algorithm
    parser_options.append(ParserOptions(
        long="motion_algorithm", 
        short="mota", 
        type=str,
        default="FourierShift", 
        nargs="+",
        choices=['OptFlow', 'BlockMotion', 'FourierShift'], 
        help="Algorithms to use for motion estimation (can specify multiple)"
    ))

    parser_options.append(ParserOptions(
        long="metrics", 
        short="m", 
        type=str, 
        # default=['mse', 'psnr'],  # Default metrics to compute
        nargs="+", 
        choices=['mse', 'psnr', 'roughness', 'ssim', 'cei', 'entropy', 'edge_preservation', 'nmse',
                 'all'], 
        help="Metrics to compute (can specify multiple)" 
    ))

    # Parse the input arguments using the defined parser options
    args = parse_input(
        parser_config=parser_options,
        prog_name="IR SBNUC prototypes"  # Program name for the parser
    )

    print(" --- Input parsed --- ")  # Indicate that input has been successfully parsed
    return args  # Return the parsed arguments


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
    sys.stdout.write(f'\r{message} -> Done! \n')

def spinner_decorator(messages):
    """Decorator to show a spinner with dynamic loading text while a function is running.

    Args:
        messages (list): List of messages to display in sequence along with the spinner.
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
        message (str): Message to display along with the loading bar.
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
