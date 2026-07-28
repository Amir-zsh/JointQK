#!/bin/bash
# RunPod container entrypoint: bring up sshd, optionally run pod_init, idle.
#
# RunPod injects PUBLIC_KEY into the environment when the user has an SSH key
# on their account; exposing TCP port 22 in the template makes the pod
# reachable via the connect tab's ssh command.
set -u

mkdir -p /root/.ssh && chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
mkdir -p /run/sshd
/usr/sbin/sshd -p "${SSH_PORT:-22}"

# POD_INIT=1 in the template env clones the repos and runs bootstrap
# automatically; without it the pod comes up bare and you drive setup over ssh.
if [ "${POD_INIT:-0}" = "1" ]; then
    /opt/runpod/pod_init.sh 2>&1 | tee /workspace/pod_init.log || true
fi

sleep infinity
