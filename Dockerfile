FROM nvcr.io/partners/salesforce/warpdrive:v1.1

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get -y install ffmpeg gedit vim tmux net-tools  \
    apt-utils git htop libgeos++-dev libproj-dev openssh-server openssh-client zsh curl

WORKDIR /workspace

# Warning: Dec 7, commands not tested.
# Following commands are not always successful, please check the output and run them manually if necessary
# oh-my-zsh installization
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" & chsh -s $(which zsh)
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
RUN sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions)/' ~/.zshrc
# tmux shortcut setup
RUN echo "bind-key s setw synchronize-panes" >> /root/.tmux.conf \
 && echo "set -g default-shell /bin/zsh" >> /root/.tmux.conf \
 && echo "set -g default-command /bin/zsh" >> /root/.tmux.conf \
 && echo "set-option -g allow-rename off" >> /root/.tmux.conf && source ~/.tmux.conf \
# conda init and zshrc sourcing
RUN conda init zsh && echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.zshrc && source ~/.zshrc
# ssh set up (currently not recommended)
# RUN mkdir /var/run/sshd && echo 'root:<yourpassword>' | chpasswd && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
# && sed -i 's/#   StrictHostKeyChecking ask/StrictHostKeyChecking no/' /etc/ssh/ssh_config && service ssh start
EXPOSE 22
# repo download and dependencies installation
RUN conda create --name mcs python=3.9 --yes && source activate mcs
RUN git clone https://github.com/movingpandas/movingpandas && cd /workspace/movingpandas && python setup.py develop && cd ..
RUN git clone https://github.com/BIT-MCS/Awesome-Mobile-Crowdsensing.git && cd Awesome-Mobile-Crowdsensing && pip install -e .
# auto boot setup
CMD ["/usr/sbin/sshd", "-D"]
CMD ["tmux", ""]