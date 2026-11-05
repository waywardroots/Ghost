"""Ghost - lets your music assistant control Ableton Live."""
from __future__ import absolute_import, print_function, unicode_literals

from .ghost import Ghost


def create_instance(c_instance):
    return Ghost(c_instance)
