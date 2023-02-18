{{ fullname | escape | underline}}

.. currentmodule:: {{ fullname }}
{% block functions %}{% if functions %}
.. rubric:: {{ _('Function Summary') }}
.. autosummary::
{% for item in functions %}
   {{ item }}
{%- endfor %}{% endif %}{% endblock %}
{% block modules %}{% if modules %}
.. rubric:: {{ _('Sub-modules') }}
.. autosummary::
   :toctree:
   :recursive:
{% endif %}{% endblock %}
{% if attributes %}
.. rubric:: {{ _('Attributes') }}
{% for item in attributes %}
.. autodata:: {{ item }}
{%- endfor %}{% endif %}
{% if classes %}
.. rubric:: {{ _('Classes') }}
{% for item in classes %}
.. autoclass:: {{ item }}
   :members:
   :undoc-members:
   :private-members:
{%- endfor %}{% endif %}
{% if exceptions %}
.. rubric:: {{ _('Exceptions') }}
{% for item in exceptions %}
.. autoclass:: {{ item }}
   :members:
   :undoc-members:
   :private-members:
{%- endfor %}{% endif %}
{% if functions %}
.. rubric:: {{ _('Functions') }}
{% for item in functions %}
.. autofunction:: {{ item }}
{%- endfor %}{% endif %}
