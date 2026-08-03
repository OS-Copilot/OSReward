"""Additional complex tasks for Pro Expense app."""

import datetime
import dataclasses
import random
from typing import Any, ClassVar
from android_world.env import device_constants
from android_world.task_evals.common_validators import sqlite_validators
from android_world.task_evals.single import expense
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.utils import datetime_utils

# Inverse of Expense.category_id_to_name (name -> category id).
_CATEGORY_NAME_TO_ID = {
    v: k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items()
}


# -----------------------------------------------------------------------------
# Complex Deletion Tasks (Filtering by Category, Amount, Note, Date)
# -----------------------------------------------------------------------------

class ExpenseDeleteFood(expense._ExpenseDeleteMultiple):
  """Task to delete all expenses in the 'Food' category."""

  complexity = 3.0
  max_steps = 20
  n_rows = 4  # Number of Food items
  n_rows_noise = 8  # Non-Food items


  @property
  def goal(self) -> str:
    return f'Delete all expenses in the "Food" category from {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Generate targets (Food)
    food_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Food')
    targets = [expense._generate_expense(category_id=food_id) for _ in range(cls.n_rows)]
    
    # Generate noise (Non-Food)
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cand.category != food_id:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteHousing(expense._ExpenseDeleteMultiple):
  """Task to delete all expenses in the 'Housing' category."""
  
  complexity = 3.0
  max_steps = 20
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return f'Delete all "Housing" related expenses from {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    housing_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Housing')
    targets = [expense._generate_expense(category_id=housing_id) for _ in range(cls.n_rows)]
    
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cand.category != housing_id:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteLowValue(expense._ExpenseDeleteMultiple):
  """Task to delete all expenses below a certain amount ($5)."""

  complexity = 3.5
  max_steps = 25
  n_rows = 4
  n_rows_noise = 8
  _THRESHOLD_CENTS = 500  # $5

  @property
  def goal(self) -> str:
    return f'Clean up {expense._APP_NAME} by deleting all small expenses under $5.00.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Targets < $5
    targets = []
    while len(targets) < cls.n_rows:
      cand = expense._generate_expense()
      # Force low amount
      cand = dataclasses.replace(cand, amount=random.randint(100, cls._THRESHOLD_CENTS - 1))
      targets.append(cand)

    # Noise >= $5
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cand.amount >= cls._THRESHOLD_CENTS:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteUrgent(expense._ExpenseDeleteMultiple):
  """Task to delete expenses with 'Urgent' in the note."""

  complexity = 3.2
  max_steps = 20
  n_rows = 3
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return f'Delete all expenses tagged as "Urgent" in {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Targets with "Urgent" note
    targets = []
    while len(targets) < cls.n_rows:
      cand = expense._generate_expense()
      cand = dataclasses.replace(cand, note="Urgent payment")
      targets.append(cand)

    # Noise without "Urgent"
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if "Urgent" not in cand.note:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeletePaidByCard(expense._ExpenseDeleteMultiple):
  """Task to delete expenses with 'Paid by card' in the note."""

  complexity = 3.2
  max_steps = 20
  n_rows = 4
  n_rows_noise = 8

  @property
  def goal(self) -> str:
    return f'Find and delete all expenses with the note "Paid by card" in {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    targets = []
    while len(targets) < cls.n_rows:
      cand = expense._generate_expense()
      cand = dataclasses.replace(cand, note="Paid by card")
      targets.append(cand)

    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if "Paid by card" not in cand.note:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteByNameSubstring(expense._ExpenseDeleteMultiple):
  """Delete expenses containing a specific string in their name (e.g., 'Taxi')."""

  complexity = 3.0
  max_steps = 20
  n_rows = 3
  n_rows_noise = 10
  _SUBSTRING = "Taxi"

  @property
  def goal(self) -> str:
    return f'Delete all "{self._SUBSTRING}" related expenses from {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Targets containing "Taxi"
    targets = []
    while len(targets) < cls.n_rows:
      cand = expense._generate_expense()
      cand = dataclasses.replace(cand, name=f"{cls._SUBSTRING} Ride {random.randint(1,10)}")
      targets.append(cand)

    # Noise NOT containing "Taxi"
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cls._SUBSTRING not in cand.name:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteByDate(expense._ExpenseDeleteMultiple):
  """Task to delete expenses created on a specific date (Oct 10, 2023)."""

  complexity = 4.0
  max_steps = 25
  n_rows = 3
  n_rows_noise = 12
  _TARGET_DAY = 10

  @property
  def goal(self) -> str:
    return f'Delete all expenses from October {self._TARGET_DAY}, 2023 in {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    target_ts = datetime_utils.create_random_october_2023_unix_ts(
        start_day=cls._TARGET_DAY, end_day=cls._TARGET_DAY
    )
    
    # Targets on Oct 10
    targets = []
    for _ in range(cls.n_rows):
      cand = expense._generate_expense(expense_unix_time_s=target_ts)
      targets.append(cand)

    # Noise NOT on Oct 10
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      cand_date = datetime.datetime.fromtimestamp(cand.created_date / 1000)
      if cand_date.day != cls._TARGET_DAY:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteEntertainmentAndSocial(expense._ExpenseDeleteMultiple):
  """Delete expenses from two categories: Entertainment and Social."""
  
  complexity = 4.0
  max_steps = 25
  n_rows = 6
  n_rows_noise = 10

  @property
  def goal(self) -> str:
    return f'Delete all "Entertainment" and "Social" expenses in {expense._APP_NAME}.'
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    ent_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Entertainment')
    soc_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Social')
    
    targets = [expense._generate_expense(category_id=random.choice([ent_id, soc_id])) for _ in range(cls.n_rows)]
    
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cand.category not in [ent_id, soc_id]:
        noise.append(cand)

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteAllIncome(expense._ExpenseDeleteMultiple):
  """Delete all Income entries."""
  
  complexity = 3.0
  max_steps = 20
  n_rows = 2
  n_rows_noise = 8
  
  @property
  def goal(self) -> str:
    return f'Delete all "Income" entries from {expense._APP_NAME}.'
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    inc_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Income')
    targets = [expense._generate_expense(category_id=inc_id) for _ in range(cls.n_rows)]
    
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cand.category != inc_id:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


# -----------------------------------------------------------------------------
# Complex Addition Tasks (Detailed, Batches, Backdated)
# -----------------------------------------------------------------------------

class ExpenseAddSpecificDate(expense._ExpenseAddMultiple):
  """Add an expense with a specific past date."""
  
  complexity = 3.5
  max_steps = 20
  n_rows = 1
  n_rows_noise = 5
  
  @property
  def goal(self) -> str:
    item = self.params[sqlite_validators.ROW_OBJECTS][0]
    dt = datetime.datetime.fromtimestamp(item.created_date / 1000)
    date_str = dt.strftime("%Y-%m-%d")
    return (
        f'In {expense._APP_NAME}, add a "{item.category_name}" expense named "{item.name}" '
        f'for ${item.amount_dollars} on {date_str}.'
    )
    
  @classmethod
  def _get_random_target_row(cls) -> sqlite_schema_utils.Expense:
    # Generate random date in Oct 2023
    ts = datetime_utils.create_random_october_2023_unix_ts(start_day=1, end_day=10)
    return expense._generate_expense(expense_unix_time_s=ts)


class ExpenseAddDetailed(expense._ExpenseAddMultiple):
  """Add a single expense with full details explicitly specified."""
  
  complexity = 3.0
  max_steps = 15
  n_rows = 1
  n_rows_noise = 5
  
  @property
  def goal(self) -> str:
    item = self.params[sqlite_validators.ROW_OBJECTS][0]
    return (
        f'Add a new expense to {expense._APP_NAME}:\n'
        f'Name: {item.name}\n'
        f'Amount: ${item.amount_dollars}\n'
        f'Category: {item.category_name}\n'
        f'Note: {item.note}'
    )


class ExpenseAddBusinessTrip(expense._ExpenseAddMultiple):
  """Add a set of business trip expenses."""
  
  complexity = 4.5
  max_steps = 30
  n_rows = 3 # Flight, Hotel, Meal
  n_rows_noise = 5
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Custom rows
    ts = expense._get_random_timestamp()
    flight = expense._generate_expense(expense_unix_time_s=ts)
    flight = dataclasses.replace(flight, name="Flight to NY", amount=45000, category=_CATEGORY_NAME_TO_ID['Transportation'], note="Business Trip")
    
    hotel = expense._generate_expense(expense_unix_time_s=ts)
    hotel = dataclasses.replace(hotel, name="Hotel Stay", amount=30000, category=_CATEGORY_NAME_TO_ID['Housing'], note="Business Trip")
    
    meal = expense._generate_expense(expense_unix_time_s=ts)
    meal = dataclasses.replace(meal, name="Client Dinner", amount=8500, category=_CATEGORY_NAME_TO_ID['Food'], note="Business Trip")
    
    targets = [flight, hotel, meal]
    noise = sqlite_schema_utils.get_random_items(cls.n_rows_noise, expense._generate_expense, replacement=False)
    
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        expense._TEXT_REPRESENTATION_TYPE: 'text_block'
    }


class ExpenseAddMonthlyBills(expense._ExpenseAddMultiple):
  """Add typical monthly bills (Rent, Internet, Electricity)."""
  
  complexity = 4.0
  max_steps = 25
  n_rows = 3
  n_rows_noise = 5

  @property
  def goal(self) -> str:
     return f"Log the monthly bills into {expense._APP_NAME}: Rent ($1200), Internet ($60), and Electricity ($45). All under 'Housing'."

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    cat_id = _CATEGORY_NAME_TO_ID['Housing']
    ts = expense._get_random_timestamp() * 1000
    
    rent = sqlite_schema_utils.Expense("Rent", 120000, cat_id, "", ts, ts)
    internet = sqlite_schema_utils.Expense("Internet", 6000, cat_id, "", ts, ts)
    electricity = sqlite_schema_utils.Expense("Electricity", 4500, cat_id, "", ts, ts)
    
    targets = [rent, internet, electricity]
    noise = sqlite_schema_utils.get_random_items(cls.n_rows_noise, expense._generate_expense, replacement=False)

    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        expense._TEXT_REPRESENTATION_TYPE: 'text_block' # Not used in custom goal but required by schema
    }


class ExpenseAddPartyExpenses(expense._ExpenseAddMultiple):
  """Add party expenses (Food, Drinks/Entertainment, Decorations/Others)."""
  
  complexity = 4.5
  max_steps = 25
  n_rows = 3
  n_rows_noise = 5
  
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    ts = expense._get_random_timestamp()
    
    e1 = dataclasses.replace(expense._generate_expense(ts), name="Party Snacks", category=_CATEGORY_NAME_TO_ID['Food'])
    e2 = dataclasses.replace(expense._generate_expense(ts), name="Drinks", category=_CATEGORY_NAME_TO_ID['Entertainment'])
    e3 = dataclasses.replace(expense._generate_expense(ts), name="Decorations", category=_CATEGORY_NAME_TO_ID['Others'])

    targets = [e1, e2, e3]
    noise = sqlite_schema_utils.get_random_items(cls.n_rows_noise, expense._generate_expense, replacement=False)
    
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        expense._TEXT_REPRESENTATION_TYPE: 'text_block'
    }


class ExpenseAddBackdated(expense._ExpenseAddMultiple):
  """Add an expense explicitly backdated to yesterday."""
  
  complexity = 3.5
  max_steps = 20
  n_rows = 1
  n_rows_noise = 5
  
  @property
  def goal(self) -> str:
    item = self.params[sqlite_validators.ROW_OBJECTS][0]
    return f'I forgot to log a "{item.category_name}" expense. Add "{item.name}" for ${item.amount_dollars} to {expense._APP_NAME}, dated yesterday (Oct 14, 2023).'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    # Fixed to Oct 14 for the "Yesterday" relative to Oct 15 env
    ts = datetime_utils.create_random_october_2023_unix_ts(start_day=14, end_day=14)
    target = expense._generate_expense(expense_unix_time_s=ts)
    noise = sqlite_schema_utils.get_random_items(cls.n_rows_noise, expense._generate_expense, replacement=False)
    
    return {
        sqlite_validators.ROW_OBJECTS: [target],
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        expense._TEXT_REPRESENTATION_TYPE: 'text_block'
    }

# -----------------------------------------------------------------------------
# Correction and Cleanup Tasks
# -----------------------------------------------------------------------------

class ExpenseDeduplicateStrict(expense._ExpenseDeleteDuplicates):
  """Strict deduplication: remove duplicates with same name, amount, category."""
  
  complexity = 3.0
  max_steps = 20
  n_rows = 1
  n_rows_noise = 20 # High noise to make searching harder
  
  @property
  def goal(self) -> str:
    return f'Find and delete any duplicate entries in {expense._APP_NAME} that match exactly in name and amount.'


class ExpenseCleanupMisc(expense._ExpenseDeleteMultiple):
  """Cleanup task: Delete all expenses in 'Others' category."""
  
  complexity = 3.0
  max_steps = 20
  n_rows = 5
  n_rows_noise = 10
  
  @property
  def goal(self) -> str:
    return f'Clear out all miscellaneous expenses (Category: "Others") from {expense._APP_NAME}.'
    
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    others_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Others')
    targets = [expense._generate_expense(category_id=others_id) for _ in range(cls.n_rows)]
    
    noise = []
    while len(noise) < cls.n_rows_noise:
      cand = expense._generate_expense()
      if cand.category != others_id:
        noise.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
    }


class ExpenseDeleteAllExceptFood(expense._ExpenseDeleteMultiple):
  """Inverse filtering: Delete everything EXCEPT a certain category."""
  
  complexity = 4.0
  max_steps = 25
  n_rows = 8 # Everything else
  n_rows_noise = 3 # Food (to keep)
  
  @property
  def goal(self) -> str:
    return f'Delete all expenses in {expense._APP_NAME} EXCEPT those in the "Food" category.'
    
  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    food_id = next(k for k, v in sqlite_schema_utils.Expense.category_id_to_name.items() if v == 'Food')
    
    # Noise = Food (Keep)
    keep = [expense._generate_expense(category_id=food_id) for _ in range(cls.n_rows_noise)]
    
    # Targets = Non-Food (Delete)
    targets = []
    while len(targets) < cls.n_rows:
      cand = expense._generate_expense()
      if cand.category != food_id:
        targets.append(cand)
        
    return {
        sqlite_validators.ROW_OBJECTS: targets,
        sqlite_validators.NOISE_ROW_OBJECTS: keep, # Noise in this validator context is what remains
    }


class ExpenseAddBudgetCorrection(expense._ExpenseAddMultiple):
  """Add an Income entry to correct a balance."""
  
  complexity = 3.0
  max_steps = 15
  n_rows = 1
  n_rows_noise = 5
  
  @property
  def goal(self) -> str:
    return f'Add an Income entry of $500.00 named "Budget Correction" to {expense._APP_NAME}.'

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    inc_id = _CATEGORY_NAME_TO_ID['Income']
    ts = expense._get_random_timestamp() * 1000
    target = sqlite_schema_utils.Expense("Budget Correction", 50000, inc_id, "Balance Fix", ts, ts)
    noise = sqlite_schema_utils.get_random_items(cls.n_rows_noise, expense._generate_expense, replacement=False)
    
    return {
        sqlite_validators.ROW_OBJECTS: [target],
        sqlite_validators.NOISE_ROW_OBJECTS: noise,
        expense._TEXT_REPRESENTATION_TYPE: 'text_block'
    }