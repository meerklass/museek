import unittest
from unittest.mock import patch, Mock

from museek.util.context_loader import ContextLoader


class TestContextLoader(unittest.TestCase):

    @patch("museek.util.context_loader.open")
    @patch("museek.util.context_loader.pickle")
    def test_ini(self, mock_pickle, mock_open):
        ContextLoader(context_path="context_path")
        mock_pickle.load.assert_called_once_with(
            mock_open.return_value.__enter__.return_value
        )
        mock_open.assert_called_once_with("context_path", "rb")

    @patch("museek.util.context_loader.open")
    @patch("museek.util.context_loader.pickle")
    def test_requirements_dict(self, mock_pickle, mock_open):
        context_loader = ContextLoader(context_path="context_path")
        mock_pickle.load.assert_called_once()
        mock_open.assert_called_once()
        mock_plugin = Mock(requirements=[Mock(), Mock()])
        requirements_dict = context_loader.requirements_dict(plugin=mock_plugin)
        expect = mock_pickle.load.return_value.__getitem__.return_value.result
        self.assertEqual(
            requirements_dict[mock_plugin.requirements[0].variable], expect
        )
        self.assertEqual(
            requirements_dict[mock_plugin.requirements[1].variable], expect
        )
