
import onecode
from flows.geomos.nullspace_main import main

def run():
    main(onecode.file_input('parameter_file', 'Parfile_paper.txt'))
