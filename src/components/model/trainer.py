from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.core import get_logger, io
from src.entity.artifact import ModelTrainerArtifact
from src.entity.config import DataTransformationConfig, ModelTrainerConfig

logger = get_logger(__name__)


class ModelTrainer(ModelTrainerConfig):
    def __init__(self):
        super().__init__()
        logger.critical('%s %s %s', '>>>' * 10, self.__class__.__name__, '<<<' * 10)
        self.transformation = DataTransformationConfig()

    def _get_train_test_data(self):
        logger.info('Loading train and test array.')
        train_arr = io.load_array(self.transformation.train_arr_path)
        test_arr = io.load_array(self.transformation.test_arr_path)

        logger.info('Splitting into X and y from train and test array.')
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        return X_train, X_test, y_train, y_test

    def _model(self, X, y):
        clf = RandomForestClassifier()
        clf.fit(X, y)
        return clf

    def _evaluate(self, model, X_train, X_test, y_train, y_test):
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        train_score = accuracy_score(y_train, y_hat_train)
        test_score = accuracy_score(y_test, y_hat_test)

        logger.info('Train score: %s', train_score)
        logger.info('Test score: %s', test_score)
        return train_score, test_score

    def _check_model_fitting(self, train_score, test_score):
        logger.info('Checking if our model is under-fit or not.')
        if test_score < self.expected_testing_score or train_score < self.expected_training_score:
            msg = (
                f'Expected training score: {self.expected_training_score} \n'
                f'Actual training score: {test_score} \n'
                f'Expected testing score: {self.expected_testing_score} \n'
                f'Actual testing score: {train_score}'
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.info('Model is not under-fit.')

        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- #

        logger.info('Checking if our model is over-fit or not.')
        diff = abs(train_score - test_score)
        logger.info('Train-Test score diff: %s', diff)
        if diff > self.overfitting_threshold:
            msg = (
                f'Train-Test score diff: {diff}\n'
                f'Overfitting threshold: {self.overfitting_threshold}'
            )
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.info('Model is not over-fit.')

    def initiate(self) -> ModelTrainerArtifact:
        X_train, X_test, y_train, y_test = self._get_train_test_data()

        logger.info('Train the model')
        model = self._model(X_train, y_train)

        train_score, test_score = self._evaluate(model, X_train, X_test, y_train, y_test)
        self._check_model_fitting(train_score, test_score)

        io.dump_model(model, self.model_path)

        return ModelTrainerArtifact(
            self.model_path,
            train_score,  # type: ignore
            test_score,  # type: ignore
        )
