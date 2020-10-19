__doc__ = """ 
Simple profiler
"""

import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

# ====BEGIN OF PROFILED PROGRAM=========
from hybrid_mc import Solution 

Solution(8,4,1,1e-2)

# =====END OF PROFILED PROGRAM==========
pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(.1)  # only 10 percent is more than enough
print(s.getvalue())
    