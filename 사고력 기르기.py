N = int(input())
B = [list(map(int, input().split())) for _ in range(N)]
D = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

for i in range(N):
    for j in range(N):
        print('*' if B[i][j] else sum(0 <= i+di < N and 0 <= j+dj < N and B[i+di][j+dj] for di, dj in D), end=' ')
    print()

