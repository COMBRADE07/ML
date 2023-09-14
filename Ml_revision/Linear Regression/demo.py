x = 121
t=x
z = str(x)
print(type(z))
rev = z[::-1]
print(rev)
t = int(rev)
print(type(t))
if t == x:
    print('palindrome')