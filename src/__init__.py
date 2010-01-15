__all__=["B1python","B1io","utils","B1macros","utils2d","guitools","fitting","xanes","asamacros","asaxseval","unstable"]
for i in __all__:
  exec('from %s import *' % i)
all1=dir()
print all1
__all__=[a for a in all1 if a[0:2]!='__']
print __all__
