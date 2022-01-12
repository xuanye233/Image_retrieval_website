#!/usr/bin/env python
from PIL import Image
import numpy
import datetime
import math
import operator
import optparse
import os
import re
import sys
import threading
import time
import webbrowser
from collections import namedtuple, OrderedDict
from functools import wraps
from getpass import getpass
from io import TextIOWrapper
from flask import g
import alipay
from flask import Flask, request, url_for, redirect
import time
from urllib.parse import urlparse, parse_qs
import json

# Py2k compat.
if sys.version_info[0] == 2:
    PY2 = True
    binary_types = (buffer, bytes, bytearray)
    decode_handler = 'replace'
    numeric = (int, long, float)
    unicode_type = unicode
    from StringIO import StringIO
else:
    PY2 = False
    binary_types = (bytes, bytearray)
    decode_handler = 'backslashreplace'
    numeric = (int, float)
    unicode_type = str
    from io import StringIO

try:
    from flask import (
        Flask, abort, escape, flash, jsonify, make_response, Markup, redirect,
        render_template, request, session, url_for)
except ImportError:
    raise RuntimeError('Unable to import flask module. Install by running '
                       'pip install flask')

try:
    from pygments import formatters, highlight, lexers
except ImportError:
    import warnings

    warnings.warn('pygments library not found.', ImportWarning)
    syntax_highlight = lambda data: '<pre>%s</pre>' % data
else:
    def syntax_highlight(data):
        if not data:
            return ''
        lexer = lexers.get_lexer_by_name('sql')
        formatter = formatters.HtmlFormatter(linenos=False)
        return highlight(data, lexer, formatter)

try:
    from peewee import __version__

    peewee_version = tuple([int(p) for p in __version__.split('.')])
except ImportError:
    raise RuntimeError('Unable to import peewee module. Install by running '
                       'pip install peewee')
else:
    if peewee_version <= (3, 0, 0):
        raise RuntimeError('Peewee >= 3.0.0 is required. Found version %s. '
                           'Please update by running pip install --update '
                           'peewee' % __version__)

from peewee import *
from peewee import IndexMetadata
from peewee import sqlite3
from playhouse.dataset import DataSet
from playhouse.migrate import migrate

CUR_DIR = os.path.realpath(os.path.dirname(__file__))
DEBUG = False
MAX_RESULT_SIZE = 1000
ROWS_PER_PAGE = 50
SECRET_KEY = 'sqlite-database-browser-0.1.0'

app = Flask(
    __name__,
    static_folder=os.path.join(CUR_DIR, 'static'),
    template_folder=os.path.join(CUR_DIR, 'templates'))
app.config.from_object(__name__)
dataset = None
migrator = None

#
# Database metadata objects.
#

TriggerMetadata = namedtuple('TriggerMetadata', ('name', 'sql'))

ViewMetadata = namedtuple('ViewMetadata', ('name', 'sql'))


#
# Database helpers.
#

class SqliteDataSet(DataSet):
    @property
    def filename(self):
        db_file = dataset._database.database
        if db_file.startswith('file:'):
            db_file = db_file[5:]
        return os.path.realpath(db_file.rsplit('?', 1)[0])

    @property
    def is_readonly(self):
        db_file = dataset._database.database
        return db_file.endswith('?mode=ro')

    @property
    def base_name(self):
        return os.path.basename(self.filename)

    @property
    def created(self):
        stat = os.stat(self.filename)
        return datetime.datetime.fromtimestamp(stat.st_ctime)

    @property
    def modified(self):
        stat = os.stat(self.filename)
        return datetime.datetime.fromtimestamp(stat.st_mtime)

    @property
    def size_on_disk(self):
        stat = os.stat(self.filename)
        return stat.st_size

    def get_indexes(self, table):
        return dataset._database.get_indexes(table)

    def get_all_indexes(self):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? ORDER BY name',
            ('index',))
        return [IndexMetadata(row[0], row[1], None, None, None)
                for row in cursor.fetchall()]

    def get_columns(self, table):
        return dataset._database.get_columns(table)

    def get_foreign_keys(self, table):
        return dataset._database.get_foreign_keys(table)

    def get_triggers(self, table):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? AND tbl_name = ?',
            ('trigger', table))
        return [TriggerMetadata(*row) for row in cursor.fetchall()]

    def get_all_triggers(self):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? ORDER BY name',
            ('trigger',))
        return [TriggerMetadata(*row) for row in cursor.fetchall()]

    def get_all_views(self):
        cursor = self.query(
            'SELECT name, sql FROM sqlite_master '
            'WHERE type = ? ORDER BY name',
            ('view',))
        return [ViewMetadata(*row) for row in cursor.fetchall()]

    def get_virtual_tables(self):
        cursor = self.query(
            'SELECT name FROM sqlite_master '
            'WHERE type = ? AND sql LIKE ? '
            'ORDER BY name',
            ('table', 'CREATE VIRTUAL TABLE%'))
        return set([row[0] for row in cursor.fetchall()])

    def get_corollary_virtual_tables(self):
        virtual_tables = self.get_virtual_tables()
        suffixes = ['content', 'docsize', 'segdir', 'segments', 'stat']
        return set(
            '%s_%s' % (virtual_table, suffix) for suffix in suffixes
            for virtual_table in virtual_tables)


#
# Flask views.
#

def convert_vector(img, w, h):
    temp = numpy.array(img)
    i = 0
    j = 0
    tem = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + temp[i][j][0]
            j = j + 1
        i = i + 1
    r1 = tem / (h * w)
    tem = 0
    i = 0
    j = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + (temp[i][j][0] - r1) * (temp[i][j][0] - r1)
            j = j + 1
        i = i + 1
    r2 = tem / (h * w)
    r2 = pow(float(r2), 1 / 2)
    tem = 0
    i = 0
    j = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + (temp[i][j][0] - r1) * (temp[i][j][0] - r1) * (temp[i][j][0] - r1)
            j = j + 1
        i = i + 1
    print(tem)
    if tem > 0:
        r3 = pow(float(tem / (h * w)), float(1 / 3))
    else:
        r3 = -pow(float(-tem / (h * w)), float(1 / 3))
    i = 0
    j = 0
    tem = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + temp[i][j][1]
            j = j + 1
        i = i + 1
    b1 = tem / (h * w)
    tem = 0
    i = 0
    j = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + (temp[i][j][1] - b1) * (temp[i][j][1] - b1)
            j = j + 1
        i = i + 1
    b2 = pow(float(tem / (h * w)), 1 / 2)
    tem = 0
    i = 0
    j = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + (temp[i][j][1] - b1) * (temp[i][j][1] - b1) * (temp[i][j][1] - b1)
            j = j + 1
        i = i + 1
    print(tem)
    if tem > 0:
        b3 = pow(float(tem / (h * w)), float(1 / 3))
    else:
        b3 = -pow(float(-tem / (h * w)), float(1 / 3))
    i = 0
    j = 0
    tem = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + temp[i][j][2]
            j = j + 1
        i = i + 1
    g1 = tem / (h * w)
    tem = 0
    i = 0
    j = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + (temp[i][j][2] - g1) * (temp[i][j][2] - g1)
            j = j + 1
        i = i + 1
    g2 = pow(float(tem / (h * w)), 1 / 2)
    tem = 0
    i = 0
    j = 0
    while i < h:
        j = 0
        while j < w:
            tem = tem + (temp[i][j][2] - g1) * (temp[i][j][2] - g1) * (temp[i][j][2] - g1)
            j = j + 1
        i = i + 1
    print(tem)
    if tem > 0:
        g3 = pow(float(tem / (h * w)), float(1 / 3))
    else:
        g3 = -pow(float(-tem / (h * w)), float(1 / 3))
    print([r1, r2, r3, b1, b2, b3, g1, g2, g3])
    return [int(r1), int(r2), int(r3), int(b1), int(b2), int(b3), int(g1), int(g2), int(g3)]


def compute(src, tem):
    temp = src[0] * tem[0] + src[1] * tem[1] + src[2] * tem[2] + src[3] * tem[3] + src[4] * tem[4] + src[5] * tem[5] + \
           src[6] * tem[6] + src[7] * tem[7] + src[8] * tem[8]
    temp = temp / pow((src[0] * src[0] + src[1] * src[1] + src[2] * src[2] + src[3] * src[3] + src[4] * src[4] + src[
        5] * src[5] + src[6] * src[6] + src[7] * src[7] + src[8] * src[8]), 1 / 2)
    temp = temp / pow((tem[0] * tem[0] + tem[1] * tem[1] + tem[2] * tem[2] + tem[3] * tem[3] + tem[4] * tem[4] + tem[
        5] * tem[5] + tem[6] * tem[6] + tem[7] * tem[7] + tem[8] * tem[8]), 1 / 2)
    print(temp)
    return temp


@app.route('/', methods=['GET', 'POST'])
def index():
    #print(g)
    global g
    if (g == True):
        return render_template('index.html', sqlite=sqlite3)
    else:
        session['authorized'] = False
        # print(session['authorized'])
        print("before")
        if request.method == 'POST':
            g = False
            f = open("./account.json", 'rb')
            file = json.load(f)
            print((str(request.form.get('form-first-name')) in file))
            if((str(request.form.get('form-first-name')) in file) == False):
                flash('The id has not been found, please check it!', 'danger')
                return render_template('sign-in.html')
            elif file[str(request.form.get('form-first-name'))] == str(request.form.get('form-last-name')) and \
                    str(request.form.get('form-email')) == "zlsnzs":
                g = True
                return redirect(session.get('next_url') or url_for('index'))
            else:
                flash('The password or license you entered is incorrect.', 'danger')
        return render_template('sign-in.html')


@app.route('/sign-up/', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        f = open("./account.json", 'rb')
        file = json.load(f)
        file[str(request.form.get('form-first-name'))] = str(request.form.get('form-last-name'))
        f = open("./account.json", 'w+', encoding="utf-8")
        jsonobj = json.dumps(file, ensure_ascii=False, indent=4)
        print(jsonobj)
        f.write(jsonobj)
        f.close()
        flash('You have successfully created your account, please log in!', 'success')
    return render_template('sign-up.html')


@app.route('/logout/', methods=['GET'])
def logout():
    session.pop('authorized', None)
    return redirect(url_for('sign-in'))


def require_table(fn):
    @wraps(fn)
    def inner(table, *args, **kwargs):
        if table not in dataset.tables:
            abort(404)
        return fn(table, *args, **kwargs)

    return inner

@app.route('/recharge/', methods=['POST','GET'])
def recharge():
    return render_template('radio-test.html')



@app.route('/recharge_result/')
def recharge_result():
    return render_template('recharge_result.html')

@app.route('/create-table/', methods=['POST'])
def table_create():
    table = (request.form.get('table_name') or '').strip()
    if not table:
        flash('Table name is required.', 'danger')
        return redirect(request.form.get('redirect') or url_for('index'))

    dataset[table]
    return redirect(url_for('table_import', table=table))


@app.route('/<table>/')
@require_table
def table_structure(table):
    ds_table = dataset[table]
    model_class = ds_table.model_class

    table_sql = dataset.query(
        'SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = ?',
        [table, 'table']).fetchone()[0]

    return render_template(
        'table_structure.html',
        columns=dataset.get_columns(table),
        ds_table=ds_table,
        foreign_keys=dataset.get_foreign_keys(table),
        indexes=dataset.get_indexes(table),
        model_class=model_class,
        table=table,
        table_sql=table_sql,
        triggers=dataset.get_triggers(table))


def get_request_data():
    if request.method == 'POST':
        return request.form
    return request.args


@app.route('/<table>/add-column/', methods=['GET', 'POST'])
@require_table
def add_column(table):
    column_mapping = OrderedDict((
        ('VARCHAR', CharField),
        ('TEXT', TextField),
        ('INTEGER', IntegerField),
        ('REAL', FloatField),
        ('BOOL', BooleanField),
        ('BLOB', BlobField),
        ('DATETIME', DateTimeField),
        ('DATE', DateField),
        ('TIME', TimeField),
        ('DECIMAL', DecimalField)))

    request_data = get_request_data()
    col_type = request_data.get('type')
    name = request_data.get('name', '')

    if request.method == 'POST':
        if name and col_type in column_mapping:
            migrate(
                migrator.add_column(
                    table,
                    name,
                    column_mapping[col_type](null=True)))
            flash('Column "%s" was added successfully!' % name, 'success')
            dataset.update_cache(table)
            return redirect(url_for('table_structure', table=table))
        else:
            flash('Name and column type are required.', 'danger')

    return render_template(
        'add_column.html',
        col_type=col_type,
        column_mapping=column_mapping,
        name=name,
        table=table)


@app.route('/<table>/drop-column/', methods=['GET', 'POST'])
@require_table
def drop_column(table):
    request_data = get_request_data()
    name = request_data.get('name', '')
    columns = dataset.get_columns(table)
    column_names = [column.name for column in columns]

    if request.method == 'POST':
        if name in column_names:
            migrate(migrator.drop_column(table, name))
            flash('Column "%s" was dropped successfully!' % name, 'success')
            dataset.update_cache(table)
            return redirect(url_for('table_structure', table=table))
        else:
            flash('Name is required.', 'danger')

    return render_template(
        'drop_column.html',
        columns=columns,
        column_names=column_names,
        name=name,
        table=table)


@app.route('/<table>/rename-column/', methods=['GET', 'POST'])
@require_table
def rename_column(table):
    request_data = get_request_data()
    rename = request_data.get('rename', '')
    rename_to = request_data.get('rename_to', '')

    columns = dataset.get_columns(table)
    column_names = [column.name for column in columns]

    if request.method == 'POST':
        if (rename in column_names) and (rename_to not in column_names):
            migrate(migrator.rename_column(table, rename, rename_to))
            flash('Column "%s" was renamed successfully!' % rename, 'success')
            dataset.update_cache(table)
            return redirect(url_for('table_structure', table=table))
        else:
            flash('Column name is required and cannot conflict with an '
                  'existing column\'s name.', 'danger')

    return render_template(
        'rename_column.html',
        columns=columns,
        column_names=column_names,
        rename=rename,
        rename_to=rename_to,
        table=table)


@app.route('/<table>/add-index/', methods=['GET', 'POST'])
@require_table
def add_index(table):
    request_data = get_request_data()
    indexed_columns = request_data.getlist('indexed_columns')
    unique = bool(request_data.get('unique'))

    columns = dataset.get_columns(table)

    if request.method == 'POST':
        if indexed_columns:
            migrate(
                migrator.add_index(
                    table,
                    indexed_columns,
                    unique))
            flash('Index created successfully.', 'success')
            return redirect(url_for('table_structure', table=table))
        else:
            flash('One or more columns must be selected.', 'danger')

    return render_template(
        'add_index.html',
        columns=columns,
        indexed_columns=indexed_columns,
        table=table,
        unique=unique)


@app.route('/<table>/drop-index/', methods=['GET', 'POST'])
@require_table
def drop_index(table):
    request_data = get_request_data()
    name = request_data.get('name', '')
    indexes = dataset.get_indexes(table)
    index_names = [index.name for index in indexes]

    if request.method == 'POST':
        if name in index_names:
            migrate(migrator.drop_index(table, name))
            flash('Index "%s" was dropped successfully!' % name, 'success')
            return redirect(url_for('table_structure', table=table))
        else:
            flash('Index name is required.', 'danger')

    return render_template(
        'drop_index.html',
        indexes=indexes,
        index_names=index_names,
        name=name,
        table=table)


@app.route('/<table>/drop-trigger/', methods=['GET', 'POST'])
@require_table
def drop_trigger(table):
    request_data = get_request_data()
    name = request_data.get('name', '')
    triggers = dataset.get_triggers(table)
    trigger_names = [trigger.name for trigger in triggers]

    if request.method == 'POST':
        if name in trigger_names:
            dataset.query('DROP TRIGGER "%s";' % name)
            flash('Trigger "%s" was dropped successfully!' % name, 'success')
            return redirect(url_for('table_structure', table=table))
        else:
            flash('Trigger name is required.', 'danger')

    return render_template(
        'drop_trigger.html',
        triggers=triggers,
        trigger_names=trigger_names,
        name=name,
        table=table)


@app.route('/<table>/content/')
@require_table
def table_content(table):
    page_number = request.args.get('page') or ''
    page_number = int(page_number) if page_number.isdigit() else 1

    dataset.update_cache(table)
    ds_table = dataset[table]
    total_rows = ds_table.all().count()
    rows_per_page = app.config['ROWS_PER_PAGE']
    total_pages = int(math.ceil(total_rows / float(rows_per_page)))
    # Restrict bounds.
    page_number = min(page_number, total_pages)
    page_number = max(page_number, 1)

    previous_page = page_number - 1 if page_number > 1 else None
    next_page = page_number + 1 if page_number < total_pages else None

    query = ds_table.all().paginate(page_number, rows_per_page)

    ordering = request.args.get('ordering')
    if ordering:
        field = ds_table.model_class._meta.columns[ordering.lstrip('-')]
        if ordering.startswith('-'):
            field = field.desc()
        query = query.order_by(field)

    field_names = ds_table.columns
    columns = [f.column_name for f in ds_table.model_class._meta.sorted_fields]

    table_sql = dataset.query(
        'SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = ?',
        [table, 'table']).fetchone()[0]

    return render_template(
        'table_content.html',
        columns=columns,
        ds_table=ds_table,
        field_names=field_names,
        next_page=next_page,
        ordering=ordering,
        page=page_number,
        previous_page=previous_page,
        query=query,
        table=table,
        total_pages=total_pages,
        total_rows=total_rows)


@app.route('/<table>/query/', methods=['GET', 'POST'])
@require_table
def table_query(table):
    data = []
    data_description = error = row_count = sql = None

    if request.method == 'POST':
        sql = request.form['sql']
        print(sql)
        if 'export_json' in request.form:
            return export(table, sql, 'json')
        elif 'export_csv' in request.form:
            return export(table, sql, 'csv')

        try:
            cursor = dataset.query(sql)
        except Exception as exc:
            error = str(exc)
        else:
            data = cursor.fetchall()[:app.config['MAX_RESULT_SIZE']]
            data_description = cursor.description
            row_count = cursor.rowcount
    else:
        if request.args.get('sql'):
            sql = request.args.get('sql')
        else:
            sql = 'SELECT *\nFROM "%s"' % (table)

    table_sql = dataset.query(
        'SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = ?',
        [table, 'table']).fetchone()[0]

    return render_template(
        'table_query.html',
        data=data,
        data_description=data_description,
        error=error,
        query_images=get_query_images(),
        row_count=row_count,
        sql=sql,
        table=table,
        table_sql=table_sql)


# search
@app.route('/<table>/search/', methods=['GET', 'POST'])
@require_table
def table_search(table):
    data = []
    data_description = error = row_count = sql = None

    if request.method == 'POST':
        name = request.form['search_name']
        year1 = request.form['year_min']
        year2 = request.form['year_max']
        sql = "select " + '*' + '\n' + "from " + table + '\n' + 'where (description like ' + "'%" + name + "%'" + \
              "or actor like " + "'%" + name + "%' " + "or director like " + "'%" + name + "%' " + "or author like " \
              + "'%" + name + "%' " + "or genre like " + "'%" + name + "%' " "or name like " + "'%" + name + "%' )" + \
              "and ( datePublished >= " + year1 + " and datePublished <= " + year2 + ")"
        print(sql)
        if 'export_json' in request.form:
            return export(table, sql, 'json')
        elif 'export_csv' in request.form:
            return export(table, sql, 'csv')

        try:
            cursor = dataset.query(sql)
        except Exception as exc:
            error = str(exc)
        else:
            data = cursor.fetchall()[:app.config['MAX_RESULT_SIZE']]
            data_description = cursor.description
            row_count = cursor.rowcount
    else:
        if request.args.get('sql'):
            sql = request.args.get('sql')
        else:
            sql = 'SELECT *\nFROM "%s"' % (table)

    table_sql = dataset.query(
        'SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = ?',
        [table, 'table']).fetchone()[0]

    return render_template(
        'table_search.html',
        data=data,
        data_description=data_description,
        error=error,
        query_images=get_query_images(),
        row_count=row_count,
        sql=sql,
        table=table,
        table_sql=table_sql)


@app.route('/table-definition/', methods=['POST'])
def set_table_definition_preference():
    key = 'show'
    show = False
    if request.form.get(key) and request.form.get(key) != 'false':
        session[key] = show = True
    elif key in session:
        del session[key]
    return jsonify({key: show})


def export(table, sql, export_format):
    model_class = dataset[table].model_class
    query = model_class.raw(sql).dicts()
    buf = StringIO()
    if export_format == 'json':
        kwargs = {'indent': 2}
        filename = '%s-export.json' % table
        mimetype = 'text/javascript'
    else:
        kwargs = {}
        filename = '%s-export.csv' % table
        mimetype = 'text/csv'

    dataset.freeze(query, export_format, file_obj=buf, **kwargs)

    response_data = buf.getvalue()
    response = make_response(response_data)
    response.headers['Content-Length'] = len(response_data)
    response.headers['Content-Type'] = mimetype
    response.headers['Content-Disposition'] = 'attachment; filename=%s' % (
        filename)
    response.headers['Expires'] = 0
    response.headers['Pragma'] = 'public'
    return response


@app.route('/<table>/import/', methods=['GET', 'POST'])
@require_table
def table_import(table):
    count = None
    request_data = get_request_data()
    strict = bool(request_data.get('strict'))

    if request.method == 'POST':
        file_obj = request.files.get('file')
        if not file_obj:
            flash('Please select an import file.', 'danger')
        elif not file_obj.filename.lower().endswith(('.csv', '.json')):
            flash('Unsupported file-type. Must be a .json or .csv file.',
                  'danger')
        else:
            if file_obj.filename.lower().endswith('.json'):
                format = 'json'
            else:
                format = 'csv'

            # Here we need to translate the file stream. Werkzeug uses a
            # spooled temporary file opened in wb+ mode, which is not
            # compatible with Python's CSV module. We'd need to reach pretty
            # far into Flask's internals to modify this behavior, so instead
            # we'll just translate the stream into utf8-decoded unicode.
            if not PY2:
                try:
                    stream = TextIOWrapper(file_obj, encoding='utf8')
                except AttributeError:
                    # The SpooledTemporaryFile used by werkzeug does not
                    # implement an API that the TextIOWrapper expects, so we'll
                    # just consume the whole damn thing and decode it.
                    # Fixed in werkzeug 0.15.
                    stream = StringIO(file_obj.read().decode('utf8'))
            else:
                stream = file_obj.stream

            try:
                with dataset.transaction():
                    count = dataset.thaw(
                        table,
                        format=format,
                        file_obj=stream,
                        strict=strict)
            except Exception as exc:
                flash('Error importing file: %s' % exc, 'danger')
            else:
                flash(
                    'Successfully imported %s objects from %s.' % (
                        count, file_obj.filename),
                    'success')
                return redirect(url_for('table_content', table=table))

    return render_template(
        'table_import.html',
        count=count,
        strict=strict,
        table=table)


@app.route('/<table>/image/', methods=['GET', 'POST'])
@require_table
def table_image(table):
    # global data, data_description, error, sql, table_sql, row_count
    data = []
    data_description = error = row_count = sql = None
    count = None
    request_data = get_request_data()
    strict = bool(request_data.get('strict'))

    if request.method == 'POST':
        file_obj = request.files.get('file')
        if not file_obj:
            flash('Please select an import file.', 'danger')
        elif not file_obj.filename.lower().endswith(('.png', '.jpg')):
            flash('Unsupported file-type. Must be a .jpg or .png file.',
                  'danger')
        else:
            if file_obj.filename.lower().endswith('.jpg'):
                format = 'jpg'
            else:
                format = 'png'
            try:
                pic = file_obj.read()
                fname = 'poster.' + format
                with open('./' + fname, 'wb') as f:
                    f.write(pic)
                # file_obj.save('')
            except Exception as exc:
                flash('Error importing file: %s' % exc, 'danger')
            else:
                flash(
                    'Successfully imported %s objects from %s.' % (
                        format, file_obj.filename),
                    'success.Start to search.')
                img = Image.open("poster.jpg")
                img = img.resize((int(img.size[0]/3), int(img.size[1]/3)), Image.ANTIALIAS)
                src_vec = convert_vector(img, img.size[0], img.size[1])
                with open("vector.txt", "r") as f:
                    id_now = ""
                    now = 0
                    count = 1
                    for line in f:  # 遍历每一行
                        wordlist = line.split()  # 将每一行的数字分开放在列表中
                        print(wordlist)
                        vect = [int(wordlist[1]), int(wordlist[2]), int(wordlist[3]), int(wordlist[4]),
                                int(wordlist[5]), int(wordlist[6]), int(wordlist[7]), int(wordlist[8]),
                                int(wordlist[9])]
                        print(vect)
                        temp = compute(src_vec, vect)
                        print(temp)
                        if temp > now:
                            now = temp
                            id_now = wordlist[0]
                        count = count + 1
                f.close()
                sql = 'select * from ' + table + ' where name like ' + '"%' + str(id_now) + '%"'
                if 'export_json' in request.form:
                    return export(table, sql, 'json')
                elif 'export_csv' in request.form:
                    return export(table, sql, 'csv')
                try:
                    cursor = dataset.query(sql)
                except Exception as exc:
                    error = str(exc)
                else:
                    data = cursor.fetchall()[:app.config['MAX_RESULT_SIZE']]
                    data_description = cursor.description
                    row_count = cursor.rowcount
    table_sql = dataset.query(
        'SELECT sql FROM sqlite_master WHERE tbl_name = ? AND type = ?',
        [table, 'table']).fetchone()[0]
    print(data)
    print(data_description)
    print(sql)
    # return redirect(url_for('table_content', table=table))
    return render_template(
        'table_image.html',
        data=data,
        data_description=data_description,
        error=error,
        query_images=get_query_images(),
        row_count=row_count,
        sql=sql,
        table=table,
        table_sql=table_sql,
        count=count,
        strict=strict,
    )


@app.route('/<table>/drop/', methods=['GET', 'POST'])
@require_table
def drop_table(table):
    if request.method == 'POST':
        model_class = dataset[table].model_class
        model_class.drop_table()
        dataset.update_cache()  # Update all tables.
        flash('Table "%s" dropped successfully.' % table, 'success')
        return redirect(url_for('index'))

    return render_template('drop_table.html', table=table)


@app.template_filter('format_index')
def format_index(index_sql):
    split_regex = re.compile(r'\bon\b', re.I)
    if not split_regex.search(index_sql):
        return index_sql

    create, definition = split_regex.split(index_sql)
    return '\nON '.join((create.strip(), definition.strip()))


@app.template_filter('value_filter')
def value_filter(value, max_length=50):
    if isinstance(value, numeric):
        return value

    if isinstance(value, binary_types):
        if not isinstance(value, (bytes, bytearray)):
            value = bytes(value)  # Handle `buffer` type.
        value = value.decode('utf-8', decode_handler)
    if isinstance(value, unicode_type):
        value = escape(value)
        if len(value) > max_length:
            return ('<span class="truncated">%s</span> '
                    '<span class="full" style="display:none;">%s</span>'
                    '<a class="toggle-value" href="#">...</a>') % (
                       value[:max_length],
                       value)
    return value


column_re = re.compile('(.+?)\((.+)\)', re.S)
column_split_re = re.compile(r'(?:[^,(]|\([^)]*\))+')


def _format_create_table(sql):
    create_table, column_list = column_re.search(sql).groups()
    columns = ['  %s' % column.strip()
               for column in column_split_re.findall(column_list)
               if column.strip()]
    return '%s (\n%s\n)' % (
        create_table,
        ',\n'.join(columns))


@app.template_filter()
def format_create_table(sql):
    try:
        return _format_create_table(sql)
    except:
        return sql


@app.template_filter('highlight')
def highlight_filter(data):
    return Markup(syntax_highlight(data))


def get_query_images():
    accum = []
    image_dir = os.path.join(app.static_folder, 'img')
    if not os.path.exists(image_dir):
        return accum
    for filename in sorted(os.listdir(image_dir)):
        basename = os.path.splitext(os.path.basename(filename))[0]
        parts = basename.split('-')
        accum.append((parts, 'img/' + filename))
    return accum


#
# Flask application helpers.
#

@app.context_processor
def _general():
    return {
        'dataset': dataset,
        'login_required': bool(app.config.get('PASSWORD')),
    }


@app.context_processor
def _now():
    return {'now': datetime.datetime.now()}


@app.before_request
def _connect_db():
    dataset.connect()


@app.teardown_request
def _close_db(exc):
    if not dataset._database.is_closed():
        dataset.close()


class PrefixMiddleware(object):
    def __init__(self, app, prefix):
        self.app = app
        self.prefix = '/%s' % prefix.strip('/')
        self.prefix_len = len(self.prefix)

    def __call__(self, environ, start_response):
        if environ['PATH_INFO'].startswith(self.prefix):
            environ['PATH_INFO'] = environ['PATH_INFO'][self.prefix_len:]
            environ['SCRIPT_NAME'] = self.prefix
            return self.app(environ, start_response)
        else:
            start_response('404', [('Content-Type', 'text/plain')])
            return ['URL does not match application prefix.'.encode()]


#
# Script options.
#

def get_option_parser():
    parser = optparse.OptionParser()
    parser.add_option(
        '-p',
        '--port',
        default=8080,
        help='Port for web interface, default=8080',
        type='int')
    parser.add_option(
        '-H',
        '--host',
        default='127.0.0.1',
        help='Host for web interface, default=127.0.0.1')
    parser.add_option(
        '-d',
        '--debug',
        action='store_true',
        help='Run server in debug mode')
    parser.add_option(
        '-x',
        '--no-browser',
        action='store_false',
        default=True,
        dest='browser',
        help='Do not automatically open browser page.')
    parser.add_option(
        '-P',
        '--password',
        action='store_true',
        dest='prompt_password',
        help='Prompt for password to access database browser.')
    parser.add_option(
        '-r',
        '--read-only',
        action='store_true',
        dest='read_only',
        help='Open database in read-only mode.')
    parser.add_option(
        '-u',
        '--url-prefix',
        dest='url_prefix',
        help='URL prefix for application.')
    return parser


def die(msg, exit_code=1):
    sys.stderr.write('%s\n' % msg)
    sys.stderr.flush()
    sys.exit(exit_code)


def open_browser_tab(host, port):
    url = 'http://%s:%s/' % (host, port)

    def _open_tab(url):
        time.sleep(1.5)
        webbrowser.open_new_tab(url)

    thread = threading.Thread(target=_open_tab, args=(url,))
    thread.daemon = True
    thread.start()


def install_auth_handler(password):
    app.config['PASSWORD'] = password

    @app.before_request
    def check_password():
        if not session.get('authorized') and request.path != '/sign-in/' and \
                not request.path.startswith(('/static/', '/favicon')):
            flash('You must log-in to view the database browser.', 'danger')
            session['next_url'] = request.base_url
            return redirect(url_for('sign-in'))


def initialize_app(filename, read_only=False, password=None, url_prefix=None):
    global dataset
    global migrator

    if password:
        install_auth_handler(password)

    if read_only:
        if sys.version_info < (3, 4, 0):
            die('Python 3.4.0 or newer is required for read-only access.')
        if peewee_version < (3, 5, 1):
            die('Peewee 3.5.1 or newer is required for read-only access.')
        db = SqliteDatabase('file:%s?mode=ro' % filename, uri=True)
        try:
            db.connect()
        except OperationalError:
            die('Unable to open database file in read-only mode. Ensure that '
                'the database exists in order to use read-only mode.')
        db.close()
        dataset = SqliteDataSet(db, bare_fields=True)
    else:
        dataset = SqliteDataSet('sqlite:///%s' % filename, bare_fields=True)

    if url_prefix:
        app.wsgi_app = PrefixMiddleware(app.wsgi_app, prefix=url_prefix)

    migrator = dataset._migrator
    dataset.close()


def main():
    # This function exists to act as a console script entry-point.
    parser = get_option_parser()
    options, args = parser.parse_args()
    if not args:
        die('Error: missing required path to database file.')

    password = None
    if options.prompt_password:
        if os.environ.get('SQLITE_WEB_PASSWORD'):
            password = os.environ['SQLITE_WEB_PASSWORD']
        else:
            while True:
                password = getpass('Enter password: ')
                password_confirm = getpass('Confirm password: ')
                if password != password_confirm:
                    print('Passwords did not match!')
                else:
                    break

    # Initialize the dataset instance and (optionally) authentication handler.
    initialize_app(args[0], options.read_only, password, options.url_prefix)

    if options.browser:
        open_browser_tab(options.host, options.port)
    # session['authorized'] = False
    print(options.host)
    app.run(host=options.host, port=options.port, debug=options.debug)


if __name__ == '__main__':
    g = False
    main()
