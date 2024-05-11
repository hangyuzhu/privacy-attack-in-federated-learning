# Federated Leakage Test

This is an experimental lib to simulate different leakage attack methods in federated learning.

## Deep Leakage Gradient

## Improved Leakage Gradient

## Robbing the Fed

One problem: the initialized model parameters of the second linear layer in ImprintBlock are too large, which could cause severe gradient explosion in client local training (model should be updated for multiple times on each client). However, if using other initialization methods or scaling to small values may deteriorate the quality of inverted images
