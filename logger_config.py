import logging
from pathlib import Path

def setup_logger(name, log_file=None):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file name. If None, uses 'app.log'
    
    Returns:
        Configured logger instance
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if log_file is None:
        log_file = "app.log"
    
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
