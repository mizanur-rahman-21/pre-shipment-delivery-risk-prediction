"""
Master Research Pipeline Entrypoint
====================================
Executable script to run the complete end-to-end research pipeline.
Delegates execution to scripts.run_full_pipeline.
"""

from scripts.run_full_pipeline import main

if __name__ == '__main__':
    main()
