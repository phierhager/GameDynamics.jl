module ValidationReports

export is_valid_report
export failure_messages
export num_failures
export pretty_validation_report
export first_failure_message

"""
Best-effort validity accessor for both simple and structured validation reports.
"""
function is_valid_report(rep)
    hasproperty(rep, :valid) || throw(ArgumentError("Object does not expose a `valid` field/property."))
    return getproperty(rep, :valid)
end

function failure_messages(rep)
    if hasproperty(rep, :issues)
        issues = getproperty(rep, :issues)
        msgs = String[]
        for iss in issues
            ok = hasproperty(iss, :ok) ? getproperty(iss, :ok) : false
            msg = hasproperty(iss, :message) ? String(getproperty(iss, :message)) : string(iss)
            ok || push!(msgs, msg)
        end
        return msgs

    elseif hasproperty(rep, :sections)
        secs = getproperty(rep, :sections)
        msgs = String[]
        for sec in secs
            sname = hasproperty(sec, :name) ? getproperty(sec, :name) : :section
            if hasproperty(sec, :issues)
                for iss in getproperty(sec, :issues)
                    ok = hasproperty(iss, :ok) ? getproperty(iss, :ok) : false
                    msg = hasproperty(iss, :message) ? String(getproperty(iss, :message)) : string(iss)
                    code = hasproperty(iss, :code) ? getproperty(iss, :code) : :issue
                    ok || push!(msgs, "[$(sname)] $(code): $(msg)")
                end
            end
        end
        return msgs

    else
        throw(ArgumentError("Unsupported validation report shape."))
    end
end

num_failures(rep) = length(failure_messages(rep))

function first_failure_message(rep)
    msgs = failure_messages(rep)
    isempty(msgs) ? nothing : first(msgs)
end

function pretty_validation_report(rep)
    lines = String[]

    family =
        hasproperty(rep, :family) ? string(getproperty(rep, :family)) :
        hasproperty(rep, :name)   ? string(getproperty(rep, :name))   :
        "validation"

    valid = is_valid_report(rep)

    push!(lines, "Validation report: " * family)
    push!(lines, "valid = " * string(valid))

    msgs = failure_messages(rep)
    if isempty(msgs)
        push!(lines, "No failures.")
    else
        push!(lines, "Failures:")
        for msg in msgs
            push!(lines, " - " * msg)
        end
    end

    return join(lines, "\n")
end

end