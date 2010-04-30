"""
fatiando.directmodels:
    A collection of direct models that can be used in the inversions
"""
__author__ = 'Leonardo Uieda (leouieda@gmail.com)'
__date__ = 'Created 16-Mar-2010'


def test(label='fast', verbose=True):
    """
    Runs the unit tests for the fatiando.directmodels package.

    Parameters:

        label: can be either 'fast' for a smaller and faster test
               or 'full' for the full test suite

        verbose: controls if the whole test information is printed
                 or just the final results
    """
    if label!='fast' and label!='full':
        from exceptions import ValueError
        raise ValueError("Test label must be either 'fast' or 'full'")

    import unittest

    import fatiando.directmodels.tests

    suite = unittest.TestSuite()
    suite.addTest(fatiando.directmodels.tests.suite(label))

    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=0)

    runner.run(suite)