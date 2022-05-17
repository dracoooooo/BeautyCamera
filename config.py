import configparser

conf = configparser.ConfigParser()
conf.optionxform = str
conf.read('config.ini')
