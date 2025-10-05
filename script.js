'use strict';

const STORAGE_KEY = 'expenses';
const DEFAULT_CATEGORIES = [
  'Food',
  'Transport',
  'Utilities',
  'Entertainment',
  'Health',
  'Shopping',
  'Bills',
  'Other'
];

function getLocalDateString(date = new Date()) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function validateExpense(candidate) {
  if (candidate == null || typeof candidate !== 'object') return false;
  const { amount, category, date, description } = candidate;
  const numericAmount = Number(amount);
  if (!Number.isFinite(numericAmount) || numericAmount <= 0) return false;
  if (typeof category !== 'string' || category.trim() === '') return false;
  if (typeof date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return false;
  if (description != null && typeof description !== 'string') return false;
  return true;
}

function normalizeExpense(expense) {
  return {
    amount: Number(expense.amount),
    category: String(expense.category),
    date: String(expense.date),
    description: expense.description ? String(expense.description) : ''
  };
}

function readExpensesFromStorage() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(validateExpense).map(normalizeExpense);
  } catch (_err) {
    return [];
  }
}

function writeExpensesToStorage(expenses) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(expenses));
}

function addExpenseToStorage(expense) {
  const normalized = normalizeExpense(expense);
  if (!validateExpense(normalized)) return;
  const existing = readExpensesFromStorage();
  existing.push(normalized);
  writeExpensesToStorage(existing);
}

function formatCurrency(amount) {
  const fixed = Number(amount || 0).toFixed(2);
  return `$${fixed}`;
}

function sortExpensesByDateDesc(expenses) {
  return [...expenses].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
}

function renderCategorySelect(selectElement) {
  selectElement.innerHTML = '';
  for (const category of DEFAULT_CATEGORIES) {
    const option = document.createElement('option');
    option.value = category;
    option.textContent = category;
    selectElement.appendChild(option);
  }
}

function renderFilterOptions(selectElement) {
  selectElement.innerHTML = '';
  const allOption = document.createElement('option');
  allOption.value = 'all';
  allOption.textContent = 'All categories';
  selectElement.appendChild(allOption);
  for (const category of DEFAULT_CATEGORIES) {
    const option = document.createElement('option');
    option.value = category;
    option.textContent = category;
    selectElement.appendChild(option);
  }
  const saved = localStorage.getItem('expenses_filter_category');
  if (saved && (saved === 'all' || DEFAULT_CATEGORIES.includes(saved))) {
    selectElement.value = saved;
  }
}

function persistSelectedFilter(categoryValue) {
  localStorage.setItem('expenses_filter_category', categoryValue);
}

function getFilteredExpenses(allExpenses, selectedCategory) {
  if (!selectedCategory || selectedCategory === 'all') return allExpenses;
  return allExpenses.filter(exp => exp.category === selectedCategory);
}

function renderTable(expensesToDisplay) {
  const tbody = document.getElementById('expenses-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const sorted = sortExpensesByDateDesc(expensesToDisplay);
  for (const expense of sorted) {
    const row = document.createElement('tr');

    const dateCell = document.createElement('td');
    dateCell.textContent = expense.date;

    const categoryCell = document.createElement('td');
    categoryCell.textContent = expense.category;

    const descriptionCell = document.createElement('td');
    descriptionCell.textContent = expense.description || '';

    const amountCell = document.createElement('td');
    amountCell.className = 'num';
    amountCell.textContent = formatCurrency(expense.amount);

    row.appendChild(dateCell);
    row.appendChild(categoryCell);
    row.appendChild(descriptionCell);
    row.appendChild(amountCell);

    tbody.appendChild(row);
  }
}

function calculateTotals(allExpenses) {
  let total = 0;
  const perCategory = new Map();
  for (const expense of allExpenses) {
    const amount = Number(expense.amount) || 0;
    total += amount;
    const existing = perCategory.get(expense.category) || 0;
    perCategory.set(expense.category, existing + amount);
  }
  return { total, perCategory };
}

function renderTotals(allExpenses) {
  const totalAmountEl = document.getElementById('total-amount');
  const categoryTotalsEl = document.getElementById('category-totals');
  if (!totalAmountEl || !categoryTotalsEl) return;

  const { total, perCategory } = calculateTotals(allExpenses);
  totalAmountEl.textContent = formatCurrency(total);

  categoryTotalsEl.innerHTML = '';
  const categories = [...perCategory.keys()].sort();
  for (const category of categories) {
    const li = document.createElement('li');
    li.textContent = `${category}: ${formatCurrency(perCategory.get(category) || 0)}`;
    categoryTotalsEl.appendChild(li);
  }
  if (categories.length === 0) {
    const li = document.createElement('li');
    li.textContent = 'No expenses yet';
    categoryTotalsEl.appendChild(li);
  }
}

function renderAll() {
  const allExpenses = readExpensesFromStorage();
  const filterSelect = document.getElementById('category-filter');
  const selectedCategory = filterSelect ? filterSelect.value : 'all';
  const filtered = getFilteredExpenses(allExpenses, selectedCategory);
  renderTable(filtered);
  renderTotals(allExpenses); // Totals are computed from all stored expenses
}

function initializeForm() {
  const form = document.getElementById('expense-form');
  const amountInput = document.getElementById('amount');
  const categorySelect = document.getElementById('category');
  const dateInput = document.getElementById('date');
  const descriptionInput = document.getElementById('description');

  if (categorySelect) renderCategorySelect(categorySelect);
  if (dateInput) dateInput.value = getLocalDateString();

  if (!form) return;
  form.addEventListener('submit', (event) => {
    event.preventDefault();

    const amount = Number(amountInput && amountInput.value ? amountInput.value : 0);
    const category = categorySelect && categorySelect.value ? categorySelect.value : '';
    const date = dateInput && dateInput.value ? dateInput.value : getLocalDateString();
    const description = descriptionInput && descriptionInput.value ? descriptionInput.value.trim() : '';

    if (!Number.isFinite(amount) || amount <= 0) {
      alert('Please enter a valid positive amount.');
      return;
    }
    if (!category) {
      alert('Please select a category.');
      return;
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      alert('Please select a valid date.');
      return;
    }

    addExpenseToStorage({ amount, category, date, description });

    form.reset();
    if (dateInput) dateInput.value = getLocalDateString();

    renderAll();
  });
}

function initializeFilter() {
  const filterSelect = document.getElementById('category-filter');
  if (!filterSelect) return;
  renderFilterOptions(filterSelect);
  filterSelect.addEventListener('change', () => {
    persistSelectedFilter(filterSelect.value);
    renderAll();
  });
}

function initialize() {
  initializeForm();
  initializeFilter();
  renderAll();
}

document.addEventListener('DOMContentLoaded', initialize);
