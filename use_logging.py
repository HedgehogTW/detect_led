import os, platform, logging
import pathlib
from time import localtime, strftime

logpath = pathlib.Path('log')
if not logpath.exists():
    logpath.mkdir() 
    print('no output log path, create one') 

fname = strftime("%Y-%m-%d-%H%M%S", localtime())
fname += '.log'
logging_file = logpath.joinpath(fname)      


print("Logging to", logging_file)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s : %(levelname)s : %(message)s',
    filename = logging_file,
    filemode = 'w',
)
logging.debug("Start of the program")
logging.info("Doing something")
logging.warning("Dying now")
