a,b = input().split()
a,b = int(a), int(b)
print("몸무게를 입력해 주세요:",a)
print("자신의 키를 입력해 주세요:",b/100)
print("bmi지수:",a/(b/100)**2)
