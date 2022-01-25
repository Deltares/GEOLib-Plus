import platform
import warnings
from ctypes import *
from multiprocessing import Pool
from os import remove
from pathlib import Path


class GefLib:
    """
    Wrapper class and functions around geflib (gef2.dll) functions required for validating of gef input files (.gef)
    Collection of helper functions for validating gef files.

    """

    def __init__(self):
        resources_dir = Path(__file__).parent / "resources"
        if platform.uname()[0] == "Windows":
            # Load DLL into memory.
            source_lib = resources_dir.joinpath("geflib.dll")
        elif platform.uname()[0] == "Linux":
            # Load SO into memory
            source_lib = resources_dir.joinpath("libgeflib.so.1")
        else:
            # name = "osx.dylib" - missing
            raise ValueError(f"Platform {platform.uname()[0]} not found")

        # Load library into memory.
        self.__lib_handle = cdll.LoadLibrary(str(source_lib))

        # Initialize DLL into memory.
        func = self.__lib_handle["init_gef"]
        func.restype = c_int
        result = func()
        if result == 1:
            return
        else:
            raise FileNotFoundError(f"{source_lib} not found")

    def __test_gef(self, component: str):
        func = self.__lib_handle["test_gef"]
        func.restype = c_int
        func.argtype = c_char_p
        return func(component.encode("utf-8"))

    def __free_gef(self):
        func = self.__lib_handle["free_gef"]
        func.restype = c_int
        return func()

    def _unload_dll(self):
        self.__free_gef()
        del self.__lib_handle

    def _read_gef(self, file_path: Path) -> int:
        func = self.__lib_handle["read_gef"]
        func.restype = c_int
        func.argtype = c_char_p
        return func(str(file_path).encode("utf-8"))

    def __get_procedurecode_flag(self) -> int:
        func = self.__lib_handle["get_procedurecode_flag"]
        func.restype = c_int
        return func()

    def __get_procedurecode_code(self) -> str:
        func = self.__lib_handle["get_procedurecode_code"]
        func.restype = c_char_p
        return func()

    def __get_reportcode_flag(self) -> int:
        func = self.__lib_handle["get_reportcode_flag"]
        func.restype = c_int
        return func()

    def __get_reportcode_code(self) -> str:
        func = self.__lib_handle["get_reportcode_code"]
        func.restype = c_char_p
        return func()

    def __get_analysiscode_flag(self) -> int:
        func = self.__lib_handle["get_analysiscode_flag"]
        func.restype = c_int
        return func()

    def __get_analysiscode_code(self) -> str:
        func = self.__lib_handle["get_analysiscode_code"]
        func.restype = c_char_p
        return func()

    def __get_measurementcode_flag(self) -> int:
        func = self.__lib_handle["get_measurementcode_flag"]
        func.restype = c_int
        return func()

    def __get_measurementcode_code(self) -> str:
        func = self.__lib_handle["get_measurementcode_code"]
        func.restype = c_char_p
        return func()

    def __get_error_level_all(self) -> int:
        func = self.__lib_handle["get_error_level_all"]
        func.restype = c_int
        return func()

    def __write_error_log(self, file_path: Path) -> int:
        func = self.__lib_handle["write_error_log"]
        func.restype = c_int
        func.argtype = c_char_p
        return func(str(file_path).encode("utf-8"))

    @staticmethod
    def get_code(code: str) -> str:
        """
        Converts Code to gef file code to procedure code.

        :param code: gef file code
        :return: gef procedure code

        """

        cs_gefbore_report: str = "GEF-BORE-REPORT"
        cs_gefcpt_report: str = "GEF-CPT-REPORT"
        cs_short_gefcpt_report: str = "CPT-REPORT"
        cs_gefzsteeninvoer: str = "GEF-ZSTEEN-INVOER"

        code = code.decode("utf-8").strip(" ").upper()

        if (code == cs_gefcpt_report) or (code == cs_short_gefcpt_report):
            return "gefCPTReport"
        if "CPT-A" in code:
            return "gefCPTAnalysis"
        if "CPT-M" in code:
            return "gefCPTMeasurement"
        if code == cs_gefbore_report:
            return "gefBoringReport"
        if "BoreA" in code:
            return "gefBoringAnalysis"
        if "BoreM" in code:
            return "gefBoringMeasurement"
        if code == cs_gefzsteeninvoer:
            return "gefWave"
        return "NotFound"

    def _get_gef_type(self) -> str:
        """
        Returns the type of the gef file. gef must have been read first

        :return: type of gef file
        :rtype str
        """

        # Try for REPORT first( for CPTs, this can be in reportcode or in procedurecode)
        flag = self.__get_procedurecode_flag()
        if flag == 1:
            return self.get_code(self.__get_procedurecode_code())

        flag = self.__get_reportcode_flag()
        if flag == 1:
            return self.get_code(self.__get_reportcode_code())

        # when GEF is not recognized as a kind of report, try for Analysis
        flag = self.__get_analysiscode_flag()
        if flag == 1:
            return self.get_code(self.__get_analysiscode_code())

        # when GEF is still not recognized, try for Measurement
        flag = self.__get_measurementcode_flag()
        if flag == 1:
            return self.get_code(self.__get_measurementcode_code())

        # when GEF is still not recognized, try for Database BORE Measurement file. To be recognized by GEF-BORE as code
        flag = self.__get_procedurecode_flag()
        if flag == 1:
            l_code = self.__get_procedurecode_code().upper()
            if "GEF-BORE" in l_code:
                return "gefBoringMeasurement"
        else:
            return "NotFound"

    def _does_gef_contain_serious_errors(self) -> (bool, bool, int):

        """
        Returns errors, warnings for reading the gef file

        :return: is there an error
        :rtype bool
        :return l_log: is a log generated (i.e. Error or Warning)
        :rtype bool
        :return l_error_level: the level of error in the gef file
        :rtype int
        """

        self.__test_gef("Header")

        l_critical_errors = [1, 2, 11, 12]
        l_non_critical_error_levels = [5, 6, 15, 16]

        l_log = False
        l_error_level = self.__get_error_level_all()
        l_error = l_error_level in l_critical_errors

        if not l_error:
            # Only continue check when nothing serious so far.
            self.__test_gef("Data")
            l_error_level = self.__get_error_level_all()
            l_error = l_error_level in l_critical_errors

        if (l_error_level > -1) and (l_error_level not in l_non_critical_error_levels):
            l_log = True

        return l_error, l_log, l_error_level

    # validation errors and warnings

    def __error_log(self, args):
        raise ValueError(
            f"Validation Error: Severe GEF Error: Code {args[0]}. See Log File: {args[1]}"
        )

    def __error_no_log(self, args):
        raise ValueError(
            f"Validation Error: Severe GEF Error: Code {args[0]}. "
            f"Check that the file conforms to GEF CPT Standard"
        )

    def __no_error_log(self, args):
        warnings.warn(
            f"Validation Warning: This file has raised a GEF warning: Code {args[0]}.  "
            f"See Log File: {args[1]}"
        )

    def __no_error_no_log(self, args):
        warnings.warn(
            f"Validation Warning: This file has raised the GEF warning: Code {args[0]}). "
            f"Check that the file conforms to GEF CPT Standard"
        )

    __valid_messages = {
        (True, True): __error_log,
        (True, False): __error_no_log,
        (False, True): __no_error_log,
        (False, False): __no_error_no_log,
    }

    def _error_handling(
        self, error: bool, log: bool, level: int, log_filename: Path = None
    ):
        if (error or log) and log_filename is not None:
            self.__write_error_log(log_filename)
        args = (str(level), str(log_filename))
        if error or log:
            self.__valid_messages[(error, log_filename is not None)](self, args)


def _validate_cpt_from_gef(filename: Path, logging: bool = True) -> int:
    """
    Raises errors or warnings and returns 0

    :param filename: path to gef file
    :type filename: Path
    :param logging: generate warning and error log file
    :type logging: bool
    :return l_error: is there an error
    :rtype bool
    :return l_log: is a log generated (i.e. Error or Warning)
    :rtype bool
    :return l_error_level: the level of error in the gef file
    :rtype int
    """

    lib_handle = GefLib()

    result = lib_handle._read_gef(filename)
    if result != 1:
        raise ValueError(
            f"{filename} unable to read file, critical error, no feedback provided. "
            f"Please check consistency of file structure"
        )

    l_err_filename = None

    if logging:
        # file name for err \ delete existing if necessary
        l_err_filename = filename.with_suffix(".err")
        try:
            # remove file if present
            remove(l_err_filename)
        except FileNotFoundError:
            pass

    gef_type = lib_handle._get_gef_type()

    # JN: I think we might only want gefCPTReport
    if gef_type in ["gefCPTReport", "gefCPTMeasurement", "gefCPTAnalysis"]:
        l_error, l_log, l_error_level = lib_handle._does_gef_contain_serious_errors()
        lib_handle._error_handling(l_error, l_log, l_error_level, l_err_filename)
    else:
        raise ValueError(f"GEF File is not of type CPT Type: {gef_type}")

    lib_handle._unload_dll()
    del lib_handle
    return 0


def validate_gef_cpt(filename: Path, logging: bool = True):
    """
    Execution of validation. This is thread safe.
    Thus multiple versions of geflib.dll can be loaded in spawned processes without interferring.

    :param filename: path to gef file
    :type filename: Path
    :param logging: generate warning and error log file
    :type logging: bool
    """
    pool = Pool(processes=1)
    results = pool.apply_async(_validate_cpt_from_gef, args=(filename, logging))
    results.get()
    pool.close()
    pool.join()
