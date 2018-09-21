import tensorflow as tf

# Usage: a class deriving from Checkpoint must call add_saveable_objects and
# add_initializable_ops after it constructs all of its graph operations.  Then,
# it must call init_vars before running the operations


def _expand_ckpt(ckpt):
    '''expand ckpt into list of ckpt files that the writer would generate'''
    suffixes = ['index', 'meta', 'data-00000-of-00001']
    return ['{}.{}'.format(ckpt, s) for s in suffixes]

class Checkpoint(object):

    def __init__(self, ckpt_path, n_keep_checkpoints, resume_step, sess):
        self.n_keep_checkpoints = n_keep_checkpoints
        self.ckpt_path = ckpt_path
        self.resume_step = resume_step
        self.sess = sess
        self.saveable_objects = {} 
        self.initializable_ops = []
        self.initialized = False
        self.saver = None


    def add_saveable_objects(self, objs):
        self.saveable_objects.update(objs)
        self.initialized = True

    def add_initializable_ops(self, ops):
        self.initializable_ops += ops


    def _maybe_init_saver(self):
        if not self.initialized:
            raise ValueError
        if self.saver is None:
            if tf.executing_eagerly():
                self.saver = tf.contrib.eager.Saver(self.saveable_objects)
            else:
                self.saver = tf.train.Saver(self.saveable_objects,
                        max_to_keep=self.n_keep_checkpoints)

    def init_vars(self):
        if tf.executing_eagerly():
            pass
        else:
            init_op = tf.variables_initializer(
                    list(self.saveable_objects.values())
                    + self.initializable_ops)
            self.sess.run(init_op)
            # self.sess.run(self.initializable_ops)

    def save(self, step):
        '''saves trainable variables, generating a special filename
        that encodes architectural parameters'''
        self._maybe_init_saver()
        if tf.executing_eagerly():
            path_pfx = self.saver.save(self.ckpt_path, step)
        else:
            path_pfx = self.saver.save(self.sess, self.ckpt_path, step)
        return path_pfx


    def restore(self):
        '''finds the appropriate checkpoint file saved by 'save' and loads into
        the existing graph'''
        from sys import stderr
        from os import access, R_OK
        ckpt_file = '{}-{}'.format(self.ckpt_path, self.resume_step)
        print('Restoring from {}'.format(ckpt_file))

        for fn in _expand_ckpt(ckpt_file):
            if not access(fn, R_OK):
                print("Couldn't find checkpoint file {}".format(fn), file=stderr)
                exit(1)
        self._maybe_init_saver()
        if tf.executing_eagerly():
            self.saver.restore(ckpt_file)
        else:
            self.saver.restore(self.sess, ckpt_file)
