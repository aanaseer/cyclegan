
class InsufficientAmount(Exception):
    pass

class Account(object):
    """
    A checking account at a local bank.  An account has the following properties:

    Attributes:
        name: a string representing the customer's name
        balance: A float holding the current balance of the customer's account
    """

    def __init__(self, name, balance=0):
        """Return an Account object whose customer name is (name) and starting balance is (balance)"""
        self.name = name
        self.balance = balance

    def withdraw(self, amount):
        """Return the balance in the account remaining after withdrawing (amount) dollars"""
        if self.balance < amount:
            raise InsufficientAmount('Not enough cash available to withdraw{}'.format(amount))
        self.balance -= amount
        return self.balance

    def deposit(self, amount):
        """Return the balance remaining in the account after depositing (amount) dollars."""
        self.balance += amount
        return self.balance