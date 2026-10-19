#!/usr/bin/env python3
"""Fail the aggregate CI gate unless every required dependency succeeded."""

import json
import os
import sys


def check(results):
    """Return unsuccessful dependencies; malformed or absent results also fail."""
    if not isinstance(results, dict) or not results:
        raise ValueError('No required CI job results were supplied')
    failed = {}
    for job, details in results.items():
        result = details.get('result') if isinstance(details, dict) else None
        if result != 'success':
            failed[job] = result or 'missing result'
    return failed


def main():
    try:
        failed = check(json.loads(os.environ['NEEDS_RESULTS_JSON']))
        if failed:
            raise ValueError('Required CI jobs did not succeed: ' + json.dumps(failed, sort_keys=True))
    except (KeyError, ValueError, TypeError) as error:
        print(error, file=sys.stderr)
        return 1
    print('All required CI jobs succeeded')
    return 0


if __name__ == '__main__':
    sys.exit(main())
