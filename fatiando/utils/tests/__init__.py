"""
Test suite for the fatiando.utils package.
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 02-Apr-2010'

import unittest


def suite(label='fast'):

    suite = unittest.TestSuite()

    return suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')