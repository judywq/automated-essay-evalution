# Automated Essay Evaluation

## Summary

This projects provides two ways of evaluating an essay.

1. NLP tools + GPT rating

Given an essay, calculate scores form NLP tools. Ask GPT for another set of scores based on certain prompot.

2. Rubric-based GPT-only rating with fine-tuning as a blackbox

Train GPT with a training set of essay and their scores (low/medium/high) based on TOFEL [Independent Writing Rubrics](./docs/toefl_writing_rubrics.pdf).

## TODO

- [ ] Usage of tools: [Coh-Metrix](https://soletlab.asu.edu/coh-metrix/)
- [ ] Framework for collecting scores


## Refs

- [Coh-Metrix](https://soletlab.asu.edu/coh-metrix/)
- 