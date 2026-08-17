class FromLog:
    """
    Class to extract specifc information from the observation log file
    """

    def __init__(self, obs_script_log: list):
        """
        :param obs_script_log: the observation log
        """
        self.obs_script_log = obs_script_log

    def straggler_list(self) -> list[str]:
        """
        Create a list of str straggler names (if any) from the observation log and return the result
        """

        element = "straggler(s):"
        straggler_list = []  # type: list[str]

        lines = [s for s in self.obs_script_log if element in s]
        for line in lines:
            if (
                "scan" in self.obs_script_log[self.obs_script_log.index(line) - 1]
                or "Slew to scan start"
                in self.obs_script_log[self.obs_script_log.index(line) - 2]
            ):
                addition = line[line.index(element) + 15 : -1].split(", ")
                for antenna_name in addition:
                    straggler_list.append(antenna_name[1:-1])

        unique_straggler_list = sorted(set(straggler_list))

        return unique_straggler_list
