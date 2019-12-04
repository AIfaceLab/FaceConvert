import multiprocessing
import time
import traceback


class SubprocessorBase(object):
    class Cli(object):
        def __init__(self, client_dict):
            self.s2c = multiprocessing.Queue()
            self.c2s = multiprocessing.Queue()
            self.p = multiprocessing.Process(
                target=self._subprocess_run, args=(client_dict,))
            self.p.start()

            self.state = None
            self.sent_time = None
            self.sent_data = None
            self.name = None
            self.host_dict = None

        def kill(self):
            self.p.terminate()
            self.p.join()
            print(self.name, ' finished')

        # overridable optional
        def get_data_name(self, data):
            # return string identificator of your 'data'
            return "undefined"

        # overridable optional
        def on_finalize(self):
            # finalize your subprocess here
            pass

        def _subprocess_run(self, client_dict):
            data = None
            s2c, c2s = self.s2c, self.c2s
            try:
                self.on_cli_initialize(client_dict)
                c2s.put({'op': 'init_ok'})
                # print(self.name, ' started')
                # get command from the server processor and then process data or close client
                while True:
                    msg = s2c.get()
                    op = msg.get('op', '')
                    if op == 'data':
                        data = msg['data']
                        result = self.process_data(data)
                        c2s.put(
                            {'op': 'success',
                             'data': data,
                             'result': result})
                        data = None  # ???
                    elif op == 'close':
                        break
                    time.sleep(0.001)

                self.on_finalize()
                c2s.put({'op': 'finalized'})
                return
            except Exception as e:
                if data is not None:
                    print('Exception while process data [%s]: %s' % (
                        self.get_data_name(data), traceback.format_exc()))
                else:
                    print('Exception: %s' % (traceback.format_exc()))
            c2s.put({'op': 'error', 'data': data})
        # overridabel

        def on_cli_initialize(self, client_dict):
            raise NotImplementedError

        # overridable
        def process_data(self, data):
            # process 'data' given from host and return result
            raise NotImplementedError

    def __init__(self, name, SubprocessorCli_class, no_response_time_sec=0):
        if not issubclass(SubprocessorCli_class, SubprocessorBase.Cli):
            raise ValueError(
                "SubprocessorCli_class must be subclass of Subprocessor.Cli")

        self.name = name
        self.SubprocessorCli_class = SubprocessorCli_class
        self.no_response_time_sec = no_response_time_sec

    # overridable
    def subprocess_info_generator(self):
        # yield per process (name, host_dict, client_dict)
        raise NotImplementedError

    # overridable
    def process_info_generator(self):
        # yield per process (name, host_dict, client_dict)
        raise NotImplementedError

    def run(self):
        self.clis = []
        # getting info about name of subprocesses, host and client dicts, and spawning them
        for name, host_dict, client_dict in self.process_info_generator():
            # try:
            cli = self.SubprocessorCli_class(client_dict)
            cli.state = 1
            cli.sent_time = time.time()
            cli.sent_data = None
            cli.name = name
            cli.host_dict = host_dict

            self.clis.append(cli)

            while True:
                # wait the c2s and just get the 'op' data
                while not cli.c2s.empty():
                    obj = cli.c2s.get()
                    op = obj.get('op', '')
                    if op == 'init_ok':
                        cli.state = 0
                    elif op == 'error':
                        cli.kill()
                        self.clis.remove(cli)
                        break
                if cli.state == 0:
                    break
            # except:
            #     raise Exception("Unable to start sunprocess %s" % (name))

        if len(self.clis) == 0:
            raise Exception(
                "Unable to start Subprocessor '%s' " % (self.name))

        # main loop of data processing
        while True:
            for cli in self.clis[:]:
                while not cli.c2s.empty():
                    obj = cli.c2s.get()
                    op = obj.get('op', '')
                    if op == 'success':
                        cli.state = 0
                    elif op == 'error':
                        cli.kill()
                        self.clis.remove(cli)

            for cli in self.clis[:]:
                if cli.state == 0:
                    data = self.get_data(cli.host_dict)
                    if data is not None:
                        cli.s2c.put({'op': 'data', 'data': data})
                        cli.sent_time = time.time()
                        cli.sent_data = data
                        cli.state = 1
                elif cli.state == 1:
                    if self.no_response_time_sec != 0 and (time.time()-cli.sent_time) > self.no_response_time_sec:
                        print('%s doesn\'t response, terminating it.' %
                              (cli.name))
                        cli.kill()
                        self.clis.remove(cli)

            if all([cli.state == 0 for cli in self.clis]):
                # all subprocesses free and no more data available to process ,ending loop
                break

        # gracefully terminating subprocesses
        for cli in self.clis[:]:
            cli.s2c.put({'op': 'close'})
            cli.sent_time = time.time()

        # if the subprocessores are finalized ,kill them
        while True:
            for cli in self.clis[:]:
                terminate_it = False
                while not cli.c2s.empty():
                    obj = cli.c2s.get()
                    obj_op = obj['op']
                    if obj_op == 'finalized':
                        terminate_it = True
                        break

                if self.no_response_time_sec != 0 and (time.time()-cli.sent_time) > self.no_response_time_sec:
                    terminate_it = True

                if terminate_it:
                    cli.state = 2
                    cli.kill()

            if all([cli.state == 2 for cli in self.clis]):
                break
            ...

    # overridable
    def get_data(self, host_dict):
        # return data for processing here
        raise NotImplementedError
