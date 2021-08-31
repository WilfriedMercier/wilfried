from wilfried.utilities import coloredMessages as cmsg

bright = cmsg.brightMessage('A bright message !')
dim    = cmsg.dimMessage(   'A dim message !')
error  = cmsg.errorMessage( 'An error message !')
ok     = cmsg.okMessage(    'A message to say everything is fine !')

for i in [bright, dim, error, ok]:
   print(i)

print('You can also combine them together')
print(bright, dim, 'A normal text in the middle.', error, ok)