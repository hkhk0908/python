class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            print(f"{amount}원이 입금되었습니다. 현재 잔액: {self.balance}원")
        else:
            print("올바른 입금 금액을 입력하세요.")

    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            print(f"{amount}원이 출금되었습니다. 현재 잔액: {self.balance}원")
        else:
            print("잔액이 부족하거나 올바른 금액이 아닙니다.")

    def display_balance(self):
        print(f"{self.owner}님의 현재 잔액: {self.balance}원")


account = BankAccount("김철수", 50000)
account.display_balance()
account.deposit(20000)
account.withdraw(10000)
account.withdraw(70000)
