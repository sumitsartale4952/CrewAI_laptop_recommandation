import os
import sys
sys.path.append('e:/AI_Project/Laptop_Recomandation/backend')
from crew import build_fallback_recommendation

profile = {'major': 'Computer Science', 'budget': 180000, 'ram': 16, 'brand': 'Any', 'os': 'Any', 'details': ''}
report = build_fallback_recommendation(profile, 'demo error')
print(report)
