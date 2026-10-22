"""Entry point of the Windows build: start the local server and open the browser."""

import multiprocessing
import sys

from logo_wizard.app.server import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    print("Logo Wizard is running. Close this window to stop it.")
    sys.exit(main())
