#!/usr/bin/env python
"""Manage the database-free Django serving layer for the built public website."""
import os
import sys


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "infiro_site.settings")
    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
