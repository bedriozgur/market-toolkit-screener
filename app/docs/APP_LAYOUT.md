APP layout notes

This app expects these relative symlinks inside the local project workspace:
- data -> $MARKET_TOOLKIT_WORKSPACE/data
- outputs -> $MARKET_TOOLKIT_WORKSPACE/outputs
- logs -> ../workspace/logs
- cache -> ../workspace/cache
- locks -> ../workspace/locks

Because the symlinks are relative, you can move the whole APP folder
to another location or another Mac and the links remain valid.
