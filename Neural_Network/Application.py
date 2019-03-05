#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File: Application.py
# @Time: 2019/3/5 19:00
# @Author: Hao Chen
# @Contact: haochen273@gmail.com

import Neural_Network
import mini_loader

training_data, validation_data, test_data = mini_loader.load_data_wrapper()

net = Neural_Network.Network([784,30,10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)