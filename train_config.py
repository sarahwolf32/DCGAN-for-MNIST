import argparse

class TrainConfig:

    def populate(self):
        # python DCGAN.py --event-file-dir test_1
        parser = argparse.ArgumentParser()
        parser.add_argument('--event-file-dir', help='Folder name for Tensorboard output', required=False)
        parser.add_argument('--log-freq', help='n, where progress is logged every n steps', required=False)
        parser.add_argument('--num-epochs', help='number of epochs to train', required=False)
        parser.add_argument('--continue_train', help='continue training where we left off', required=False)
        args = parser.parse_args()
        self._populate_from_args(args)

    def _populate_from_args(self, args):

        # logging
        self.event_filename = args.event_file_dir or 'summary'
        self.log_freq = args.log_freq or 2

        # training
        self.num_epochs = args.num_epochs or 15

        # saving/restoring
        self.should_continue = args.continue_train or False
        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_freq = 5