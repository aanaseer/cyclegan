import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(os.path.join(ROOT, 'src'))

import pytest
from experiments import Account, InsufficientAmount

def test_setting_name():
    jack_account = Account(name = "Jack")
    assert jack_account.name == "Jack"

def test_default_balance():
    jack_account = Account(name = "Jack")
    assert jack_account.balance == 0

def test_setting_balance():
    jill_account = Account(name = "Jill", balance = 250)
    assert jill_account.balance == 250

def test_account_deposit():
    jill_account = Account(name = "Jill", balance = 250)
    jill_account.deposit(120)
    assert jill_account.balance == 370

def test_account_withdraw():
    jill_account = Account(name = "Jill", balance = 250)
    jill_account.withdraw(10)
    assert wallet.balance == 240

def test_account_withdraw_raises_exception_on_insufficient_amount():
    jack_account = Account(name = "Jack")
    with pytest.raises(InsufficientAmount):
        jack_account.withdraw(100)