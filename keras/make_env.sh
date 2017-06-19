#!/usr/bin/env bash
apt-get update && apt-get upgrade -y
apt-get install gcc
apt-get install git
apt-get install linux-headers-$(uname -r)