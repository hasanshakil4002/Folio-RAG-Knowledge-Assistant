import sys

import uvicorn

if sys.version_info < (3, 12):
    sys.exit("Python 3.12 or newer is required.")

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
