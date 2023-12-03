import re

text = '{"score": 0.0, "reasoning": "\'The ab"cd\nef"}'
reasoning_pattern = re.compile(r'"reasoning":\s*"(.*)"', re.DOTALL)
reasoning_pattern.findall(text)[0]
