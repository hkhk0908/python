A = ["0","1","2","3","4","5","6","7","8","9"]
B = ["A","B","C","D","E","F","G","H","I","J"]
C = []

C, i = [], 0
for c in range(1, 11):
    C += A[i:i+c] + B[i:i+c]
    i += c

print("LIST A:", A)
print("LIST B:", B)
print("LIST C:", C)