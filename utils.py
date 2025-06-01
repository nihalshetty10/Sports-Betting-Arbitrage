def american_to_prob(odds):
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def format_date(dt):
    return dt.strftime('%Y-%m-%d')
