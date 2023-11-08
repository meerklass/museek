class FromLog:
    def __init__(self, obs_script_log: list):
        self.obs_script_log = obs_script_log

    def straggler_list(self) -> list[str]:
        """
        Create a straggler list (if any) reading from the observation log
        """

        element = 'straggler(s):'
        straggler_list = []
        unique_straggler_list = []

        lines = [s for s in self.obs_script_log if element in s]
        for line in lines:
            if 'scan' in self.obs_script_log[self.obs_script_log.index(line)-1] \
                or 'Slew to scan start' in self.obs_script_log[self.obs_script_log.index(line)-2]:
                addition = line[line.index(element)+15:-1].split(', ')
                for ant in addition:
                    straggler_list.append(ant[1:-1])

                unique_straggler_list = sorted(set(straggler_list))

        return unique_straggler_list
